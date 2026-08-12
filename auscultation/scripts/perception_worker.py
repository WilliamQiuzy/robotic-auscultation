"""Non-blocking, latest-frame-only perception worker for the Linbox console.

The Orbbec capture thread remains the sole camera owner.  It submits immutable
``CameraFrame`` snapshots here; a slower GPU forward pass may drop intermediate
preview frames, but it can never stall acquisition or diagnostic recording.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class AssetIntegrityError(RuntimeError):
    """A model/runtime asset does not match its release manifest."""


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path, expected_sha256, *, label):
    path = Path(path)
    if not _valid_sha256(expected_sha256):
        raise AssetIntegrityError(f"{label} expected SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        raise AssetIntegrityError(f"{label} must be a regular non-symlink file: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssetIntegrityError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def _failed_result(frame_index, reason, error=None):
    result = {
        "frame_index": int(frame_index),
        "back_present": False,
        "initialized": False,
        "detector_confidence": 0.0,
        "reason": str(reason),
        "points": [],
    }
    if error:
        result["error"] = str(error)
    return result


class LatestFrameWorker:
    """Run one processor in a daemon thread with a one-slot overwrite queue."""

    def __init__(self, processor, *, name="wam-perception"):
        self._processor = processor
        self._name = name
        self._condition = threading.Condition()
        self._pending = None
        self._latest = _failed_result(0, "MODEL_STARTING")
        self._thread = None
        self._stopping = False
        self._last_submitted_index = -1
        self._last_completed_index = -1
        self._last_completed_monotonic = None
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._error = None
        self._last_inference_ms = None
        self._last_stage_timings = None

    def start(self):
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stopping = False
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self


    def submit(self, frame):
        index = int(getattr(frame, "index"))
        with self._condition:
            if self._stopping or index <= self._last_submitted_index:
                return False
            self._last_submitted_index = index
            self._submitted += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = frame
            self._condition.notify()
        return True

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                result = self._processor(frame)
                if not isinstance(result, dict):
                    raise TypeError("perception processor must return a dict")
                result = dict(result)
                result.setdefault("frame_index", int(frame.index))
                result.setdefault("camera_host_t_s", float(frame.t))
                result.setdefault("points", [])
                if not isinstance(result["points"], list):
                    result["points"] = list(result["points"])
                error = None
            except Exception as exc:  # fail closed: stale anatomy is never retained
                result = _failed_result(frame.index, "MODEL_ERROR", str(exc))
                result["camera_host_t_s"] = float(getattr(frame, "t", 0.0))
                error = str(exc)
            completed = time.monotonic()
            inference_ms = (completed - started) * 1000.0
            result["inference_ms"] = inference_ms
            with self._condition:
                self._latest = result
                self._processed += 1
                self._last_completed_index = int(frame.index)
                self._last_completed_monotonic = completed
                self._last_inference_ms = inference_ms
                timings = result.get("stage_timings_ms")
                if isinstance(timings, dict):
                    self._last_stage_timings = dict(timings)
                self._error = error
                self._condition.notify_all()

    def latest(self):
        with self._condition:
            result = self._latest
        # Results contain only JSON-like values; a shallow copy protects top-level metadata.
        return dict(result)

    def health(self):
        with self._condition:
            thread_alive = self._thread is not None and self._thread.is_alive()
            age = (
                None if self._last_completed_monotonic is None
                else max(0.0, time.monotonic() - self._last_completed_monotonic)
            )
            error = self._error
            return {
                "status": "ERROR" if error else ("OK" if thread_alive else "STOPPED"),
                "error": error,
                "thread_alive": thread_alive,
                "submitted": self._submitted,
                "processed": self._processed,
                "dropped_before_inference": self._dropped,
                "last_submitted_index": self._last_submitted_index,
                "last_completed_index": self._last_completed_index,
                "pending": self._pending is not None,
                "last_result_age_s": age,
                "last_inference_ms": self._last_inference_ms,
                "last_stage_timings_ms": self._last_stage_timings,
            }

    def drain(self, timeout=2.0):
        """Wait until the item last accepted at call entry has completed.

        Callers must stop concurrent submissions before invoking this method if
        they need a hard capture boundary.  The worker remains usable after a
        successful drain.
        """
        try:
            timeout_value = None if timeout is None else float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("drain timeout must be non-negative or None") from exc
        if timeout_value is not None and (not math.isfinite(timeout_value) or timeout_value < 0):
            raise ValueError("drain timeout must be non-negative or None")
        with self._condition:
            target = self._last_submitted_index
            if target < 0 or self._last_completed_index >= target:
                return True
            deadline = None if timeout_value is None else time.monotonic() + timeout_value
            while self._last_completed_index < target:
                if self._stopping:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def is_alive(self):
        """True iff the worker thread exists and has not yet exited."""
        with self._condition:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def stop(self, timeout=2.0):
        """Signal stop and join; return True iff the thread actually exited.

        A stuck processor (e.g. a hung GPU forward pass) cannot be forced to
        return, so ``join`` may time out with the thread still alive.  Reporting
        that failure lets the caller refuse to spawn a replacement worker on top
        of a live thread (which would silently grow the thread count and race on
        the shared camera stream).
        """
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()


class BoundedFifoWorker:
    """Non-blocking, loss-intolerant FIFO for camera-rate capture writes.

    Unlike :class:`LatestFrameWorker`, this worker never overwrites an accepted
    item. A full queue rejects the new frame visibly; callers must abort the
    active recording. Explicit intervals provide hard Start/End boundaries.
    """

    def __init__(self, processor, *, capacity, name="wam-capture-fifo"):
        self._processor = processor
        self._name = str(name)
        self._capacity = int(capacity)
        if self._capacity <= 0:
            raise ValueError("FIFO capacity must be positive")
        self._condition = threading.Condition()
        self._queue = deque()
        self._thread = None
        self._stopping = False
        self._interval_active = False
        self._inflight_index = None
        self._last_accepted_index = -1
        self._first_accepted_index = None
        self._last_completed_index = -1
        self._accepted = 0
        self._processed = 0
        self._overflow_rejections = 0
        self._outside_interval_rejections = 0
        self._non_monotonic_rejections = 0
        self._abandoned_after_error = 0
        self._max_queue_depth = 0
        self._error = None

    def start(self):
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread.start()
        return self

    def begin_interval(self):
        with self._condition:
            if (
                self._stopping
                or self._interval_active
                or self._inflight_index is not None
                or self._queue
            ):
                return False
            self._interval_active = True
            self._last_accepted_index = -1
            self._first_accepted_index = None
            self._last_completed_index = -1
            self._accepted = 0
            self._processed = 0
            self._overflow_rejections = 0
            self._outside_interval_rejections = 0
            self._non_monotonic_rejections = 0
            self._abandoned_after_error = 0
            self._max_queue_depth = 0
            self._error = None
            return True

    def close_interval(self):
        with self._condition:
            self._interval_active = False
            self._condition.notify_all()
            if self._last_accepted_index < 0:
                return None
            return self._last_accepted_index

    def submit(self, item):
        index = int(getattr(item, "index"))
        with self._condition:
            if not self._interval_active or self._stopping:
                self._outside_interval_rejections += 1
                return False
            if self._error is not None:
                return False
            if self._last_accepted_index >= 0 and index <= self._last_accepted_index:
                self._non_monotonic_rejections += 1
                return False
            if len(self._queue) >= self._capacity:
                self._overflow_rejections += 1
                return False
            self._queue.append(item)
            if self._first_accepted_index is None:
                self._first_accepted_index = index
            self._last_accepted_index = index
            self._accepted += 1
            self._max_queue_depth = max(self._max_queue_depth, len(self._queue))
            self._condition.notify()
            return True

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                item = self._queue.popleft()
                index = int(getattr(item, "index"))
                self._inflight_index = index
            try:
                self._processor(item)
            except Exception as exc:  # capture failure invalidates the interval
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._abandoned_after_error += len(self._queue)
                    self._queue.clear()
                    self._interval_active = False
                    self._inflight_index = None
                    self._condition.notify_all()
                continue
            with self._condition:
                self._processed += 1
                self._last_completed_index = index
                self._inflight_index = None
                self._condition.notify_all()

    def drain(self, *, through_index=None, timeout=2.0):
        try:
            timeout_value = None if timeout is None else float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("drain timeout must be non-negative or None") from exc
        if timeout_value is not None and (
            not math.isfinite(timeout_value) or timeout_value < 0
        ):
            raise ValueError("drain timeout must be non-negative or None")
        with self._condition:
            target = self._last_accepted_index if through_index is None else through_index
            if target is None or int(target) < 0:
                return True
            target = int(target)
            deadline = None if timeout_value is None else time.monotonic() + timeout_value
            while self._last_completed_index < target:
                if self._error is not None or self._stopping:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def health(self):
        with self._condition:
            alive = self._thread is not None and self._thread.is_alive()
            return {
                "status": "ERROR" if self._error else ("OK" if alive else "STOPPED"),
                "error": self._error,
                "thread_alive": alive,
                "interval_active": self._interval_active,
                "queue_capacity": self._capacity,
                "queue_depth": len(self._queue),
                "max_queue_depth": self._max_queue_depth,
                "inflight_index": self._inflight_index,
                "accepted": self._accepted,
                "processed": self._processed,
                "first_accepted_index": self._first_accepted_index,
                "last_accepted_index": self._last_accepted_index,
                "last_completed_index": self._last_completed_index,
                "overflow_rejections": self._overflow_rejections,
                "outside_interval_rejections": self._outside_interval_rejections,
                "non_monotonic_rejections": self._non_monotonic_rejections,
                "abandoned_after_error": self._abandoned_after_error,
                # Compatibility for the UI: capture FIFO has no overwrite mode.
                "dropped_before_inference": 0,
            }

    def is_alive(self):
        """True iff the worker thread exists and has not yet exited."""
        with self._condition:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def stop(self, timeout=2.0):
        """Signal stop and join; return True iff the thread actually exited."""
        with self._condition:
            self._interval_active = False
            self._stopping = True
            self._abandoned_after_error += len(self._queue)
            self._queue.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()


def tracker_result_to_dict(output):
    """Convert the production tracker's runtime adapter into JSON-safe data."""
    def finite_vector(value, size):
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return (
            result
            if len(result) == size and all(math.isfinite(item) for item in result)
            else None
        )

    points = []
    for point in getattr(output, "points", ()):
        pixel = getattr(point, "pixel", None)
        xyz = getattr(point, "xyz_camera", None)
        if pixel is None:
            continue
        pixel_values = [float(value) for value in pixel]
        if len(pixel_values) != 2 or not all(math.isfinite(value) for value in pixel_values):
            continue
        xyz_values = None
        if xyz is not None:
            xyz_values = finite_vector(xyz, 3)
        xyz_patient = finite_vector(getattr(point, "xyz_patient", None), 3)
        measurement_residual = float(
            getattr(point, "measurement_residual_m", float("inf"))
        )
        robot_occluded = bool(getattr(point, "robot_occluded", False))
        direct_geometry_accepted = bool(
            getattr(point, "direct_geometry_accepted", False)
            and xyz_values is not None
            and xyz_patient is not None
            and math.isfinite(measurement_residual)
            and not robot_occluded
        )
        points.append({
            "name": str(getattr(point, "name")),
            "pixel": pixel_values,
            "xyz_camera": xyz_values,
            "status": str(getattr(point, "status")),
            "measured": bool(getattr(point, "measured", False)),
            "position_std": float(getattr(point, "position_std", float("inf"))),
            "xyz_patient": xyz_patient,
            "measurement_residual_m": (
                measurement_residual
                if math.isfinite(measurement_residual)
                else None
            ),
            "robot_occluded": robot_occluded,
            "direct_geometry_accepted": direct_geometry_accepted,
        })
    support = [
        bool(value) for value in getattr(output, "pose_support_mask", ())
    ]
    rotation_rows = getattr(
        output, "sim3_rotation_camera_from_patient", None,
    )
    rotation = None
    try:
        candidate_rows = [
            finite_vector(row, 3) for row in rotation_rows
        ]
        if len(candidate_rows) == 3 and all(row is not None for row in candidate_rows):
            rotation = candidate_rows
    except TypeError:
        pass
    translation = finite_vector(
        getattr(output, "sim3_translation_camera_from_patient_m", None), 3,
    )
    scale = float(getattr(output, "sim3_scale", float("nan")))
    uncertainty = float(
        getattr(output, "pose_uncertainty_bound_m", float("inf"))
    )
    geometry_accepted = bool(
        getattr(output, "pose_geometry_accepted", False)
        and rotation is not None
        and translation is not None
        and math.isfinite(scale)
        and scale > 0.0
        and math.isfinite(uncertainty)
    )
    return {
        "back_present": bool(getattr(output, "back_present", False)),
        "initialized": bool(getattr(output, "initialized", False)),
        "detector_confidence": float(getattr(output, "detector_confidence", 0.0)),
        "reason": str(getattr(output, "reason", "MODEL_ERROR")),
        "pose_measured_this_frame": bool(
            getattr(output, "pose_measured_this_frame", False)
        ),
        "pose_support_mask": support,
        "pose_geometry_accepted": geometry_accepted,
        "pose_uncertainty_bound_m": (
            uncertainty if math.isfinite(uncertainty) else None
        ),
        "sim3_rotation_camera_from_patient": rotation,
        "sim3_scale": scale if math.isfinite(scale) and scale > 0.0 else None,
        "sim3_translation_camera_from_patient_m": translation,
        "points": points,
    }


class TrackerProcessor:
    """Adapter from camera evidence to the live JSON contract.

    The original ``CameraFrame`` adapter remains available for offline and
    robot-free callers.  A caller that requires robot masking is production
    safety mode: it must inject the complete independent back-mask, metric
    patient-pose and occlusion-runtime chain and submit a ``FrameEvidenceBundle``.
    """

    def __init__(
        self,
        tracker,
        runtime_frame_fn,
        *,
        presence_gate=None,
        presence_reset_after_misses=3,
        require_robot_mask=False,
        back_mask_provider=None,
        patient_pose_provider=None,
        occlusion_runtime=None,
        episode_id=None,
    ):
        if type(presence_reset_after_misses) is not int or presence_reset_after_misses <= 0:
            raise ValueError("presence_reset_after_misses must be a positive integer")
        if type(require_robot_mask) is not bool:
            raise ValueError("require_robot_mask must be a bool")
        evidence_chain = (
            back_mask_provider,
            patient_pose_provider,
            occlusion_runtime,
        )
        configured_count = sum(value is not None for value in evidence_chain)
        if configured_count not in {0, len(evidence_chain)}:
            raise ValueError(
                "production occlusion evidence chain must be injected completely"
            )
        self._production_evidence_chain = configured_count == len(evidence_chain)
        if require_robot_mask and not self._production_evidence_chain:
            raise ValueError(
                "require_robot_mask needs the complete production occlusion "
                "evidence chain"
            )
        if require_robot_mask and presence_gate is None:
            raise ValueError(
                "require_robot_mask needs a same-frame operator presence gate"
            )
        if self._production_evidence_chain:
            back_builder = getattr(back_mask_provider, "build", None)
            if not callable(back_builder) and not callable(back_mask_provider):
                raise ValueError("back_mask_provider must build a BackMaskPacket")
            if not callable(getattr(patient_pose_provider, "process_evidence", None)):
                raise ValueError(
                    "patient_pose_provider must expose process_evidence"
                )
            if not callable(getattr(occlusion_runtime, "advance", None)):
                raise ValueError("occlusion_runtime must expose advance")
            if not callable(getattr(occlusion_runtime, "reset", None)):
                raise ValueError("occlusion_runtime must expose reset")
            point_names = tuple(getattr(occlusion_runtime, "point_names", ()))
            if (
                not point_names
                or len(set(point_names)) != len(point_names)
                or any(not isinstance(name, str) or not name for name in point_names)
            ):
                raise ValueError("occlusion_runtime point names are invalid")
            if not (
                (isinstance(episode_id, str) and bool(episode_id))
                or callable(episode_id)
            ):
                raise ValueError(
                    "production occlusion evidence chain requires an episode id"
                )
        self.tracker = tracker
        self.runtime_frame_fn = runtime_frame_fn
        self.presence_gate = presence_gate
        self.presence_reset_after_misses = presence_reset_after_misses
        self.require_robot_mask = require_robot_mask
        self.back_mask_provider = back_mask_provider
        self.patient_pose_provider = patient_pose_provider
        self.occlusion_runtime = occlusion_runtime
        self._episode_id_source = episode_id
        self._active_episode_id = None
        self._safety_reset_fault = None
        self._consecutive_presence_misses = 0
        self._authorized_confirmation_identity = None

    def __call__(self, item):
        if self._production_evidence_chain:
            return self._process_production_evidence(item)
        return self._process_legacy_frame(item)

    def _process_legacy_frame(self, item):
        frame = getattr(item, "camera_frame", item)
        mask_packet = getattr(item, "robot_mask_packet", None)
        bundle_id = getattr(item, "bundle_id", None)
        if (
            getattr(frame, "status", None) != "OK"
            or getattr(frame, "frame", None) is None
            or getattr(frame, "depth", None) is None
        ):
            return _failed_result(frame.index, "CAMERA_NOT_READY")
        robot_mask = None
        if mask_packet is not None:
            if (
                int(getattr(mask_packet, "frame_index", -1)) != int(frame.index)
                or getattr(mask_packet, "mask", None) is None
            ):
                return _failed_result(frame.index, "ROBOT_MASK_FRAME_MISMATCH")
            robot_mask = mask_packet.mask
        elif self.require_robot_mask:
            return _failed_result(frame.index, "ROBOT_MASK_UNAVAILABLE")
        gate_result = None
        if self.presence_gate is not None:
            gate_result = self.presence_gate.assess(frame)
            if not isinstance(gate_result, dict):
                raise TypeError("posterior presence gate must return a dict")
            if int(gate_result.get("frame_index", frame.index)) != int(frame.index):
                raise ValueError("posterior presence gate returned a different frame index")
            publication_authorized = bool(
                gate_result.get("posterior_present") is True
                and gate_result.get("operator_confirmed") is True
            )
            if not publication_authorized:
                # Display evidence is fail-closed on every negative frame, but
                # the private temporal track survives a short run of pose-model
                # misses.  This avoids a visible full reacquisition for one
                # transient YOLO miss while still making it impossible to show
                # a stale constellation: this result always contains no points.
                qualified_people = gate_result.get("qualified_people")
                single_person = bool(
                    gate_result.get("person_present") is True
                    and type(qualified_people) is int
                    and qualified_people == 1
                )
                if single_person:
                    # Keep the commissioned landmark model executing whenever
                    # exactly one person is visible, while strictly separating
                    # model health from authorization to publish anatomy.  The
                    # private result is reduced to coordinate-free telemetry.
                    self._consecutive_presence_misses = 0
                    raw_probe = self.tracker.process(
                        frame.frame,
                        frame.depth,
                        t=float(frame.t),
                        depth_scale=float(frame.depth_scale),
                        robot_mask=robot_mask,
                    )
                    probe = tracker_result_to_dict(
                        self.runtime_frame_fn(raw_probe)
                    )
                    result = _failed_result(
                        frame.index,
                        gate_result.get("reason", "BACK_CONFIRMATION_REQUIRED"),
                    )
                    result["landmark_probe"] = {
                        "executed": True,
                        "frame_index": int(frame.index),
                        "raw_initialized": bool(probe.get("initialized") is True),
                        "raw_point_count": len(tuple(probe.get("points", ()))),
                        "raw_reason": str(probe.get("reason", "MODEL_ERROR")),
                    }
                    result["posterior_gate"] = dict(gate_result)
                    if bundle_id is not None:
                        result["frame_evidence_bundle_id"] = str(bundle_id)
                    if mask_packet is not None:
                        result["robot_mask"] = self._robot_mask_metadata(mask_packet)
                    return result
                self._consecutive_presence_misses += 1
                if self._consecutive_presence_misses == self.presence_reset_after_misses:
                    self.tracker.reset("NO_BACK")
                    self._authorized_confirmation_identity = None
                result = _failed_result(frame.index, gate_result.get(
                    "reason", "NO_POSTERIOR_PERSON",
                ))
                result["posterior_gate"] = dict(gate_result)
                return result
            confirmation = gate_result.get("back_confirmation")
            confirmation = confirmation if isinstance(confirmation, dict) else {}
            identity = (
                gate_result.get("confirmation_frame_index"),
                confirmation.get("confirmation_preview_frame_index"),
            )
            if any(type(value) is not int or value < 0 for value in identity):
                raise ValueError("confirmed posterior gate identity is invalid")
            if identity != self._authorized_confirmation_identity:
                # A private probe may have observed a front-facing or OOD
                # person.  Never promote that canonical state: a new explicit
                # back confirmation always starts a fresh 12-frame tracker init.
                self.tracker.reset("BACK_CONFIRMATION_ACTIVATED")
                self._authorized_confirmation_identity = identity
            self._consecutive_presence_misses = 0
        raw = self.tracker.process(
            frame.frame,
            frame.depth,
            t=float(frame.t),
            depth_scale=float(frame.depth_scale),
            robot_mask=robot_mask,
        )
        result = tracker_result_to_dict(self.runtime_frame_fn(raw))
        if bundle_id is not None:
            result["frame_evidence_bundle_id"] = str(bundle_id)
        if mask_packet is not None:
            result["robot_mask"] = self._robot_mask_metadata(mask_packet)
        if gate_result is not None:
            result["posterior_gate"] = dict(gate_result)
        return result

    @staticmethod
    def _finite_vector(value, size):
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return np.full(size, np.nan, dtype=np.float64)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            return np.full(size, np.nan, dtype=np.float64)
        return vector

    def _reset_production_state(self, reason, *, reset_patient=True):
        """Clear every state that could otherwise republish stale geometry."""

        errors = []
        for target, method_name, args in (
            (self.occlusion_runtime, "reset", ()),
            (
                self.patient_pose_provider,
                "reset",
                (),
            ) if reset_patient else (None, "reset", ()),
            (self.tracker, "reset", (str(reason),)),
        ):
            if target is None:
                continue
            method = getattr(target, method_name, None)
            if not callable(method):
                continue
            try:
                method(*args)
            except TypeError:
                # Some legacy landmark trackers expose reset() without a
                # reason.  This is safe because the reset itself remains hard.
                try:
                    method()
                except Exception as exc:  # noqa: BLE001 - fail-closed reset
                    errors.append(f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - fail-closed reset
                errors.append(f"{type(exc).__name__}: {exc}")
        self._consecutive_presence_misses = 0
        return "; ".join(errors) if errors else None

    def _production_failure(self, frame_index, reason, error=None):
        reset_error = self._reset_production_state(reason)
        self._safety_reset_fault = reset_error
        detail = str(error) if error is not None else None
        if reset_error:
            detail = (
                f"{detail}; safety reset failed: {reset_error}"
                if detail else f"safety reset failed: {reset_error}"
            )
        return _failed_result(frame_index, reason, detail)

    def _resolve_episode_id(self):
        value = (
            self._episode_id_source()
            if callable(self._episode_id_source)
            else self._episode_id_source
        )
        if not isinstance(value, str) or not value:
            raise ValueError("episode id is unavailable")
        if self._active_episode_id is not None and value != self._active_episode_id:
            reset_error = self._reset_production_state("EPISODE_CHANGED")
            if reset_error:
                raise RuntimeError(f"episode reset failed: {reset_error}")
        self._active_episode_id = value
        return value

    def set_episode_id(self, episode_id):
        """Change a fixed episode identity and atomically clear all track state."""

        if callable(self._episode_id_source):
            raise RuntimeError("episode id is controlled by the injected provider")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode id must be a nonempty string")
        if episode_id != self._episode_id_source:
            reset_error = self._reset_production_state("EPISODE_CHANGED")
            if reset_error:
                raise RuntimeError(f"episode reset failed: {reset_error}")
            self._episode_id_source = episode_id
            self._active_episode_id = episode_id

    def _build_back_mask(self, camera_frame):
        builder = getattr(self.back_mask_provider, "build", None)
        operator_gate = None
        gate_result = None
        if self.presence_gate is not None:
            from ausc.perception.tracking.posterior_back_mask_provider import (
                OperatorConfirmedBackGate,
            )

            gate_result = self.presence_gate.assess(camera_frame)
            if not isinstance(gate_result, dict):
                raise TypeError("posterior presence gate must return a dict")
            if int(gate_result.get("frame_index", -1)) != int(camera_frame.index):
                raise ValueError(
                    "posterior presence gate returned a different frame index"
                )
            posterior_present = gate_result.get("posterior_present", False)
            operator_confirmed = gate_result.get("operator_confirmed", False)
            if type(posterior_present) is not bool:
                raise ValueError("posterior gate presence must be a bool")
            if type(operator_confirmed) is not bool:
                raise ValueError("posterior gate confirmation must be a bool")
            person_box = None
            if posterior_present and operator_confirmed:
                raw_box = self._finite_vector(gate_result.get("box"), 4)
                if not np.isfinite(raw_box).all():
                    raise ValueError(
                        "confirmed posterior gate person box is invalid"
                    )
                height, width = camera_frame.frame.shape[:2]
                x0 = max(0, min(width, int(math.floor(float(raw_box[0])))))
                y0 = max(0, min(height, int(math.floor(float(raw_box[1])))))
                x1 = max(0, min(width, int(math.ceil(float(raw_box[2])))))
                y1 = max(0, min(height, int(math.ceil(float(raw_box[3])))))
                person_box = (x0, y0, x1, y1)
            operator_gate = OperatorConfirmedBackGate.for_frame(
                camera_frame=camera_frame,
                posterior_present=posterior_present,
                operator_confirmed=operator_confirmed,
                person_box_xyxy=person_box,
            )
        if callable(builder):
            packet = (
                builder(camera_frame=camera_frame, gate=operator_gate)
                if operator_gate is not None
                else builder(camera_frame=camera_frame)
            )
        else:
            packet = (
                self.back_mask_provider(camera_frame, operator_gate)
                if operator_gate is not None
                else self.back_mask_provider(camera_frame)
            )
        return packet, operator_gate, gate_result

    @staticmethod
    def _robot_mask_metadata(robot):
        return {
            "frame_index": int(robot.frame_index),
            "mask_sha256": str(robot.mask_sha256),
            "geometry_release_id": str(robot.geometry_release_id),
            "tool_release_id": str(robot.tool_release_id),
            "handeye_release_id": str(robot.handeye_release_id),
            "handeye_sha256": str(getattr(robot, "handeye_sha256", "")),
            "handeye_capability_schema": str(getattr(
                robot, "handeye_capability_schema", "",
            )),
            "handeye_calibration_tier": str(getattr(
                robot, "handeye_calibration_tier", "",
            )),
            "handeye_policy_sha256": str(getattr(
                robot, "handeye_policy_sha256", "",
            )),
            "handeye_calibration_status": str(getattr(
                robot, "handeye_calibration_status", "COMPLETE",
            )),
            "handeye_provisional": bool(getattr(
                robot, "handeye_provisional", False,
            )),
            "handeye_quality_gate_passed": getattr(
                robot, "handeye_quality_gate_passed", None,
            ),
            "handeye_workspace_1sigma_max_m": getattr(
                robot, "handeye_workspace_1sigma_max_m", None,
            ),
            "handeye_covariance_bound_transform": str(getattr(
                robot,
                "handeye_covariance_bound_transform",
                "transform_camera_from_follower_base",
            )),
            "handeye_covariance_scope": str(getattr(
                robot, "handeye_covariance_scope", "legacy_production",
            )),
            "handeye_systematic_bias_status": str(getattr(
                robot,
                "handeye_systematic_bias_status",
                "LEGACY_NOT_APPLICABLE",
            )),
            "handeye_systematic_bias_bound_m": getattr(
                robot, "handeye_systematic_bias_bound_m", None,
            ),
            "handeye_autonomous_targeting_authorized": bool(getattr(
                robot, "handeye_autonomous_targeting_authorized", True,
            )),
            "handeye_provisional_policy_allowance_m": float(getattr(
                robot, "handeye_provisional_policy_allowance_m", 0.0,
            )),
        }

    @staticmethod
    def _evidence_metadata(
        evidence,
        patient_pose_packet=None,
        *,
        operator_gate=None,
        gate_result=None,
    ):
        bundle = evidence.frame_bundle
        back = evidence.back_mask_packet
        robot = bundle.robot_mask_packet
        metadata = {
            "frame_evidence_bundle_id": str(bundle.bundle_id),
            "patient_frame_evidence_id": str(evidence.evidence_id),
            "back_mask_packet_id": str(back.packet_id),
            "back_mask": {
                "frame_index": int(back.frame_index),
                "mask_sha256": str(back.mask_sha256),
                "depth_sha256": str(back.depth_sha256),
                "operator_gate_decision_id": str(
                    back.operator_gate_decision_id
                ),
                "release_id": str(back.back_mask_release_id),
                "model_sha256": str(back.back_mask_model_sha256),
                "status": str(back.status),
            },
            "robot_mask": TrackerProcessor._robot_mask_metadata(robot),
        }
        if patient_pose_packet is not None:
            metadata["patient_pose"] = {
                "frame_index": int(getattr(patient_pose_packet, "frame_index", -1)),
                "valid": bool(getattr(patient_pose_packet, "valid", False)),
                "reason": str(getattr(patient_pose_packet, "reason", "INVALID")),
                "episode_id": str(
                    getattr(patient_pose_packet, "episode_id", "unbound")
                ),
            }
        if operator_gate is not None:
            metadata["operator_back_gate_id"] = str(operator_gate.decision_id)
            metadata["operator_back_gate"] = {
                "frame_index": int(operator_gate.frame_index),
                "posterior_present": bool(operator_gate.posterior_present),
                "operator_confirmed": bool(operator_gate.operator_confirmed),
                "person_box_xyxy": (
                    None
                    if operator_gate.person_box_xyxy is None
                    else [int(value) for value in operator_gate.person_box_xyxy]
                ),
                "reason": str(
                    (gate_result or {}).get("reason", "OPERATOR_GATE_BOUND")
                ),
            }
        return metadata

    @staticmethod
    def _with_metadata(result, metadata):
        result.update(metadata)
        return result

    def _landmark_observation(self, landmark_frame, evidence):
        from ausc.perception.tracking.occlusion_runtime import (
            OcclusionFrameObservation,
            OcclusionPointObservation,
        )

        points = []
        for point in tuple(getattr(landmark_frame, "points", ())):
            position_std = float(getattr(point, "position_std", float("inf")))
            explicit_bound = getattr(point, "uncertainty_bound_m", None)
            uncertainty_bound = (
                float(explicit_bound)
                if explicit_bound is not None
                else position_std * math.sqrt(7.814727903251179)
            )
            points.append(OcclusionPointObservation(
                name=str(getattr(point, "name", "")),
                xyz_camera_m=self._finite_vector(
                    getattr(point, "xyz_camera", None), 3,
                ),
                xyz_patient_m=self._finite_vector(
                    getattr(point, "xyz_patient", None), 3,
                ),
                pixel=self._finite_vector(getattr(point, "pixel", None), 2),
                measured=bool(getattr(point, "measured", False)),
                direct_geometry_accepted=bool(
                    getattr(point, "direct_geometry_accepted", False)
                ),
                robot_occluded=bool(
                    getattr(point, "robot_occluded", False)
                ),
                uncertainty_bound_m=uncertainty_bound,
            ))
        camera = evidence.frame_bundle.camera_frame
        return OcclusionFrameObservation(
            frame_index=int(camera.index),
            camera_host_t_s=float(camera.t),
            camera_id=str(camera.camera_id),
            camera_profile_sha256=str(camera.camera_profile_sha256),
            aligned_depth_shape=tuple(int(value) for value in camera.depth.shape),
            back_present=bool(
                evidence.back_mask_packet.status == "OK"
                and getattr(landmark_frame, "back_present", False)
            ),
            initialized=bool(getattr(landmark_frame, "initialized", False)),
            points=tuple(points),
        )

    def _safe_result_to_dict(
        self,
        safe_frame,
        *,
        landmark_frame,
        patient_pose_packet,
        metadata,
    ):
        expected_names = tuple(self.occlusion_runtime.point_names)
        safe_points = tuple(getattr(safe_frame, "points", ()))
        if (
            int(getattr(safe_frame, "frame_index", -1))
            != int(metadata["back_mask"]["frame_index"])
            or float(getattr(safe_frame, "camera_host_t_s", float("nan")))
            != float(metadata.get("camera_host_t_s", float("nan")))
        ):
            return None, "OCCLUSION_RESULT_FRAME_MISMATCH"
        if not (
            bool(getattr(safe_frame, "back_present", False))
            and bool(getattr(safe_frame, "initialized", False))
            and str(getattr(safe_frame, "reason", "")) == "TRACKING"
        ):
            result = _failed_result(
                getattr(safe_frame, "frame_index", 0),
                getattr(safe_frame, "reason", "OCCLUSION_NOT_READY"),
            )
            result["detector_confidence"] = float(
                getattr(landmark_frame, "detector_confidence", 0.0)
            )
            return self._with_metadata(result, metadata), None
        names = tuple(str(getattr(point, "name", "")) for point in safe_points)
        if (
            len(safe_points) != len(expected_names)
            or set(names) != set(expected_names)
            or len(set(names)) != len(names)
            or any(not bool(getattr(point, "valid", False)) for point in safe_points)
        ):
            return None, "OCCLUSION_RESULT_INCOMPLETE"

        serialized = []
        for point in safe_points:
            robot_occluded = bool(getattr(point, "robot_occluded", False))
            if robot_occluded and not (
                str(getattr(point, "occlusion_state", ""))
                == "PATIENT_FRAME_LOCKED"
                and str(getattr(point, "source", ""))
                == "independent_patient_pose"
                and bool(getattr(point, "measured", True)) is False
                and bool(getattr(patient_pose_packet, "valid", False))
            ):
                return None, "UNSAFE_OCCLUDED_POINT_SOURCE"
            xyz = self._finite_vector(getattr(point, "xyz_camera_m", None), 3)
            pixel = self._finite_vector(getattr(point, "pixel", None), 2)
            bound = float(
                getattr(point, "uncertainty_bound_m", float("inf"))
            )
            if not np.isfinite(xyz).all() or not np.isfinite(pixel).all() or not (
                math.isfinite(bound) and bound >= 0.0
            ):
                return None, "OCCLUSION_RESULT_INVALID_POINT"
            patient_xyz = None
            patient_from_camera = getattr(
                patient_pose_packet, "patient_from_camera", None,
            )
            if callable(patient_from_camera) and bool(
                getattr(patient_pose_packet, "valid", False)
            ):
                try:
                    patient = self._finite_vector(patient_from_camera(xyz), 3)
                    if np.isfinite(patient).all():
                        patient_xyz = [float(value) for value in patient]
                except (TypeError, ValueError):
                    patient_xyz = None
            serialized.append({
                "name": str(point.name),
                "pixel": [float(value) for value in pixel],
                "xyz_camera": [float(value) for value in xyz],
                "xyz_patient": patient_xyz,
                "status": str(point.occlusion_state),
                "occlusion_state": str(point.occlusion_state),
                "valid": True,
                "measured": bool(point.measured),
                "position_std": bound,
                "uncertainty_bound_m": bound,
                "robot_occluded": robot_occluded,
                "source": str(point.source),
                "reason": str(point.reason),
                "direct_geometry_accepted": bool(
                    str(point.occlusion_state) == "DIRECT"
                    and point.measured
                    and not robot_occluded
                ),
                "measurement_residual_m": None,
            })
        result = {
            "frame_index": int(safe_frame.frame_index),
            "camera_host_t_s": float(safe_frame.camera_host_t_s),
            "back_present": True,
            "initialized": True,
            "detector_confidence": float(
                getattr(landmark_frame, "detector_confidence", 0.0)
            ),
            "reason": "TRACKING",
            "pose_measured_this_frame": bool(
                getattr(patient_pose_packet, "valid", False)
            ),
            "points": serialized,
        }
        return self._with_metadata(result, metadata), None

    def _process_production_evidence(self, item):
        from ausc.perception.robot.frame_evidence_bundle import (
            FrameEvidenceBundle,
            FrameEvidenceBundleError,
        )
        from ausc.perception.tracking.back_mask_packet import BackMaskPacket
        from ausc.perception.tracking.patient_frame_evidence import (
            PatientFrameEvidence,
            PatientFrameEvidenceError,
        )

        try:
            fallback_index = int(getattr(item, "index", 0))
        except (TypeError, ValueError, OverflowError):
            fallback_index = 0
        if self._safety_reset_fault is not None:
            reset_error = self._reset_production_state("SAFETY_RESET_RETRY")
            if reset_error:
                self._safety_reset_fault = reset_error
                return _failed_result(
                    fallback_index,
                    "SAFETY_RESET_FAILED",
                    f"safety reset failed: {reset_error}",
                )
            self._safety_reset_fault = None
        if not isinstance(item, FrameEvidenceBundle):
            return self._production_failure(
                fallback_index, "FRAME_EVIDENCE_BUNDLE_REQUIRED",
            )
        try:
            item.__post_init__()
        except (FrameEvidenceBundleError, TypeError, ValueError) as exc:
            return self._production_failure(
                fallback_index, "FRAME_EVIDENCE_BUNDLE_INVALID", exc,
            )
        camera = item.camera_frame
        stage_ms = {}
        _stage_t0 = time.monotonic()
        try:
            back_packet, operator_gate, gate_result = self._build_back_mask(camera)
        except Exception as exc:  # noqa: BLE001 - provider is an evidence boundary
            return self._production_failure(
                camera.index, "BACK_MASK_PROVIDER_ERROR", exc,
            )
        stage_ms["back_mask_and_gate"] = (time.monotonic() - _stage_t0) * 1000.0
        if back_packet is None:
            return self._production_failure(
                camera.index, "BACK_MASK_UNAVAILABLE",
            )
        if not isinstance(back_packet, BackMaskPacket):
            return self._production_failure(
                camera.index, "BACK_MASK_PACKET_INVALID",
            )
        try:
            back_frame_mismatch = bool(
                int(getattr(back_packet, "frame_index", -1))
                != int(camera.index)
                or int(getattr(back_packet, "camera_host_monotonic_ns", -1))
                != int(camera.camera_host_monotonic_ns)
                or str(getattr(back_packet, "camera_id", ""))
                != str(camera.camera_id)
                or str(getattr(back_packet, "camera_profile_sha256", ""))
                != str(camera.camera_profile_sha256)
                or str(getattr(back_packet, "alignment_sha256", ""))
                != str(camera.alignment_sha256)
                or str(getattr(back_packet, "rgb_sha256", ""))
                != str(camera.rgb_sha256)
                or str(getattr(back_packet, "depth_sha256", ""))
                != str(camera.depth_sha256)
                or operator_gate is None
                or str(getattr(back_packet, "operator_gate_decision_id", ""))
                != str(operator_gate.decision_id)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return self._production_failure(
                camera.index, "BACK_MASK_PACKET_INVALID", exc,
            )
        if back_frame_mismatch:
            return self._production_failure(
                camera.index, "BACK_MASK_FRAME_MISMATCH",
            )
        _stage_t0 = time.monotonic()
        try:
            evidence = PatientFrameEvidence.build(
                frame_bundle=item,
                back_mask_packet=back_packet,
            )
        except PatientFrameEvidenceError as exc:
            return self._production_failure(
                camera.index, "BACK_MASK_PACKET_INVALID", exc,
            )
        stage_ms["evidence_build"] = (time.monotonic() - _stage_t0) * 1000.0
        try:
            episode_id = self._resolve_episode_id()
        except Exception as exc:  # noqa: BLE001 - injected identity boundary
            return self._production_failure(
                camera.index, "EPISODE_ID_UNAVAILABLE", exc,
            )
        _stage_t0 = time.monotonic()
        try:
            patient_pose_packet = self.patient_pose_provider.process_evidence(
                evidence, episode_id,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed pose evidence
            return self._production_failure(
                camera.index, "PATIENT_POSE_EVIDENCE_ERROR", exc,
            )
        stage_ms["patient_pose"] = (time.monotonic() - _stage_t0) * 1000.0
        if patient_pose_packet is None:
            return self._production_failure(
                camera.index, "PATIENT_POSE_PACKET_UNAVAILABLE",
            )
        try:
            patient_frame_index = int(
                getattr(patient_pose_packet, "frame_index", -1)
            )
            patient_frame_t = float(
                getattr(patient_pose_packet, "camera_host_t_s", float("nan"))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return self._production_failure(
                camera.index, "PATIENT_POSE_PACKET_INVALID", exc,
            )
        if (
            patient_frame_index != int(camera.index)
            or patient_frame_t != float(camera.t)
        ):
            return self._production_failure(
                camera.index, "PATIENT_POSE_FRAME_MISMATCH",
            )
        metadata = self._evidence_metadata(
            evidence,
            patient_pose_packet,
            operator_gate=operator_gate,
            gate_result=gate_result,
        )
        metadata["camera_host_t_s"] = float(camera.t)
        if back_packet.status != "OK":
            result = self._production_failure(camera.index, "NO_BACK")
            return self._with_metadata(result, metadata)

        _stage_t0 = time.monotonic()
        try:
            raw = self.tracker.process(
                evidence.rgb,
                evidence.aligned_depth,
                t=float(evidence.t),
                depth_scale=float(camera.depth_scale),
                robot_mask=evidence.robot_mask,
            )
            landmark_frame = self.runtime_frame_fn(raw)
        except Exception as exc:  # noqa: BLE001 - model output is untrusted
            return self._production_failure(
                camera.index, "LANDMARK_MODEL_ERROR", exc,
            )
        stage_ms["landmark_model"] = (time.monotonic() - _stage_t0) * 1000.0
        landmark_index = getattr(landmark_frame, "frame_index", camera.index)
        try:
            landmark_index = int(landmark_index)
        except (TypeError, ValueError, OverflowError) as exc:
            return self._production_failure(
                camera.index, "LANDMARK_RESULT_INVALID", exc,
            )
        if landmark_index != int(camera.index):
            return self._production_failure(
                camera.index, "LANDMARK_FRAME_MISMATCH",
            )
        _stage_t0 = time.monotonic()
        try:
            observation = self._landmark_observation(landmark_frame, evidence)
            safe_frame = self.occlusion_runtime.advance(
                observation,
                robot_mask_packet=item.robot_mask_packet,
                patient_pose_packet=patient_pose_packet,
            )
        except Exception as exc:  # noqa: BLE001 - safety runtime boundary
            return self._production_failure(
                camera.index, "OCCLUSION_RUNTIME_ERROR", exc,
            )
        try:
            result, unsafe_reason = self._safe_result_to_dict(
                safe_frame,
                landmark_frame=landmark_frame,
                patient_pose_packet=patient_pose_packet,
                metadata=metadata,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return self._production_failure(
                camera.index, "OCCLUSION_RESULT_INVALID", exc,
            )
        stage_ms["occlusion_runtime"] = (time.monotonic() - _stage_t0) * 1000.0
        if unsafe_reason is not None:
            return self._production_failure(
                camera.index, unsafe_reason,
            )
        result["stage_timings_ms"] = {
            key: round(float(value), 1) for key, value in stage_ms.items()
        }
        return result
