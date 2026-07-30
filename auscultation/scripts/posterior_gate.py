"""Fail-closed person presence and operator-confirmed back gate.

The auscultation landmark checkpoint always emits ten heatmap peaks, including
for out-of-distribution backgrounds.  It therefore cannot also be the evidence
that a patient is present.  A separately pinned YOLO model is used only to
answer whether exactly one sufficiently large person is visible.  It is not
trusted to infer front versus back: the fixed clinical camera often cannot see
a face and rear-view COCO pose keypoints are weak.

Until a dedicated back-present model is trained and commissioned, an operator
must explicitly confirm that the single visible person faces away from the
camera.  That confirmation is bound to the observed person box and fails closed
on multiple people, target change, or sustained target loss.

The gate never creates auscultation points.  The v9 NormalsDetector remains the
sole point source; this result merely permits or resets that tracker on the same
camera frame.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


class PosteriorGateError(RuntimeError):
    """The independent human/posterior gate could not make a trusted decision."""


class BackConfirmationError(RuntimeError):
    """An operator confirmation is absent, stale, or changed during Start."""


def _finite_array(value, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


def select_single_person(
    detections: Iterable[Mapping[str, object]],
    *,
    image_shape: Sequence[int],
    min_box_confidence: float = 0.20,
    min_box_area_fraction: float = 0.01,
) -> dict:
    """Return one box-backed person decision without inferring orientation.

    The thresholds deliberately operate on the detector box only.  Face,
    shoulder and hip keypoints are ignored because they are not dependable in
    the deployed rear-view camera geometry.
    """

    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("image_shape must contain positive height and width") from exc
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    confidence_floor = float(min_box_confidence)
    area_floor = float(min_box_area_fraction)
    if (
        not math.isfinite(confidence_floor)
        or not 0.0 <= confidence_floor <= 1.0
        or not math.isfinite(area_floor)
        or not 0.0 < area_floor <= 1.0
    ):
        raise ValueError("person box thresholds are invalid")

    image_area = float(height * width)
    detected = 0
    invalid = 0
    candidates = []
    for raw in detections:
        if not isinstance(raw, Mapping):
            invalid += 1
            continue
        box = _finite_array(raw.get("box"), (4,))
        try:
            box_confidence = float(raw.get("box_confidence"))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if box is None or not math.isfinite(box_confidence):
            invalid += 1
            continue
        detected += 1
        box_width = max(0.0, float(box[2] - box[0]))
        box_height = max(0.0, float(box[3] - box[1]))
        area_fraction = box_width * box_height / image_area
        if box_confidence < confidence_floor or area_fraction < area_floor:
            continue
        candidates.append({
            "box": [float(value) for value in box],
            "box_confidence": box_confidence,
            "box_area_fraction": area_fraction,
        })

    base = {
        "detected_people": detected,
        "scene_people": len(candidates),
        "qualified_people": len(candidates),
        "invalid_detections": invalid,
        "frame_shape": [height, width],
    }
    if invalid:
        return {
            "person_present": False,
            "reason": "PERSON_DETECTION_INVALID",
            **base,
        }
    if not candidates:
        return {
            "person_present": False,
            "reason": "NO_PERSON",
            **base,
        }
    if len(candidates) != 1:
        return {
            "person_present": False,
            "reason": "AMBIGUOUS_PEOPLE",
            **base,
        }
    return {
        "person_present": True,
        "reason": "ONE_PERSON",
        **base,
        **candidates[0],
    }


def _boxes_match(
    reference: Sequence[float],
    observed: Sequence[float],
    *,
    frame_shape: Sequence[int],
    min_iou: float,
    max_center_shift_fraction: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    ref = _finite_array(reference, (4,))
    obs = _finite_array(observed, (4,))
    if ref is None or obs is None:
        return False
    ref_width, ref_height = ref[2] - ref[0], ref[3] - ref[1]
    obs_width, obs_height = obs[2] - obs[0], obs[3] - obs[1]
    ref_area = float(ref_width * ref_height)
    obs_area = float(obs_width * obs_height)
    if ref_width <= 0 or ref_height <= 0 or obs_width <= 0 or obs_height <= 0:
        return False
    ratio = obs_area / ref_area
    if not float(min_area_ratio) <= ratio <= float(max_area_ratio):
        return False

    left = max(float(ref[0]), float(obs[0]))
    top = max(float(ref[1]), float(obs[1]))
    right = min(float(ref[2]), float(obs[2]))
    bottom = min(float(ref[3]), float(obs[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = ref_area + obs_area - intersection
    iou = 0.0 if union <= 0 else intersection / union

    try:
        height, width = float(frame_shape[0]), float(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return False
    diagonal = math.hypot(width, height)
    if not math.isfinite(diagonal) or diagonal <= 0:
        return False
    ref_center = ((ref[0] + ref[2]) / 2.0, (ref[1] + ref[3]) / 2.0)
    obs_center = ((obs[0] + obs[2]) / 2.0, (obs[1] + obs[3]) / 2.0)
    center_shift = math.hypot(
        float(obs_center[0] - ref_center[0]),
        float(obs_center[1] - ref_center[1]),
    ) / diagonal
    return bool(
        iou >= float(min_iou)
        and center_shift <= float(max_center_shift_fraction)
    )


class OperatorConfirmedBackGate:
    """Bind one explicit back confirmation to a continuous single-person track."""

    def __init__(
        self,
        person_gate,
        *,
        max_missed_frames: int = 5,
        max_observation_age_s: float = 1.25,
        max_preview_lag_frames: int = 30,
        min_iou: float = 0.20,
        max_center_shift_fraction: float = 0.20,
        min_area_ratio: float = 0.35,
        max_area_ratio: float = 3.0,
        clock=time.monotonic,
    ) -> None:
        if type(max_missed_frames) is not int or max_missed_frames <= 0:
            raise ValueError("max_missed_frames must be a positive integer")
        if type(max_preview_lag_frames) is not int or max_preview_lag_frames <= 0:
            raise ValueError("max_preview_lag_frames must be a positive integer")
        values = (
            max_observation_age_s, min_iou, max_center_shift_fraction,
            min_area_ratio, max_area_ratio,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("back confirmation thresholds must be finite and positive")
        if float(min_area_ratio) > float(max_area_ratio):
            raise ValueError("back confirmation area ratio bounds are invalid")
        self.person_gate = person_gate
        self.max_missed_frames = max_missed_frames
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_preview_lag_frames = max_preview_lag_frames
        self.min_iou = float(min_iou)
        self.max_center_shift_fraction = float(max_center_shift_fraction)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self._clock = clock
        self._lock = threading.RLock()
        self._latest_decision = None
        self._latest_observed_at = None
        self._active = False
        self._current = False
        self._confirmed_box = None
        self._confirmed_shape = None
        self._confirmation_frame_index = None
        self._confirmation_preview_frame_index = None
        self._activation_after_frame_index = None
        self._missed_frames = 0
        self._reason = "BACK_CONFIRMATION_REQUIRED"
        self._validity_epoch = 0

    def _clear_locked(self, reason: str) -> None:
        self._validity_epoch += 1
        self._active = False
        self._current = False
        self._confirmed_box = None
        self._confirmed_shape = None
        self._confirmation_frame_index = None
        self._confirmation_preview_frame_index = None
        self._activation_after_frame_index = None
        self._missed_frames = 0
        self._reason = str(reason)

    def _observation_age_locked(self, now: float) -> float:
        if self._latest_observed_at is None:
            return float("inf")
        return max(0.0, float(now) - float(self._latest_observed_at))

    def _expire_stale_locked(self, now: float) -> float:
        age = self._observation_age_locked(now)
        if age > self.max_observation_age_s:
            if self._active or self._current:
                self._clear_locked("CONFIRMATION_OBSERVATION_STALE")
            elif self._latest_decision is not None:
                self._reason = "PERSON_OBSERVATION_STALE"
        return age

    def _mark_not_current_locked(self) -> None:
        if self._current:
            self._validity_epoch += 1
        self._current = False

    def clear_confirmation(self, reason="BACK_CONFIRMATION_REQUIRED") -> dict:
        with self._lock:
            self._clear_locked(str(reason))
            return self._snapshot_locked()

    clear = clear_confirmation

    def _snapshot_locked(self, *, now=None) -> dict:
        now = float(self._clock()) if now is None else float(now)
        age = self._expire_stale_locked(now)
        latest = self._latest_decision or {}
        confirmable = bool(
            latest.get("person_present") is True
            and age <= self.max_observation_age_s
        )
        return {
            "required": True,
            "mode": "operator",
            "status": (
                "ACTIVE" if self._active and self._current else
                "CONFIRMED_PENDING_FRAME" if self._active else
                "CONFIRMABLE" if confirmable else "UNCONFIRMED"
            ),
            "confirmable": confirmable,
            "confirmed": bool(self._active and self._current),
            "active": bool(self._active),
            "reason": str(self._reason),
            "person_frame_index": latest.get("frame_index"),
            "confirmation_frame_index": self._confirmation_frame_index,
            "confirmation_preview_frame_index": self._confirmation_preview_frame_index,
            "observation_age_s": None if not math.isfinite(age) else age,
            "missed_frames": int(self._missed_frames),
        }

    def confirmation_snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    snapshot = confirmation_snapshot

    def confirm_back(self, *, preview_frame_index) -> dict:
        now = float(self._clock())
        with self._lock:
            self._expire_stale_locked(now)
            decision = self._latest_decision
            age = self._observation_age_locked(now)
            if (
                not isinstance(decision, dict)
                or decision.get("person_present") is not True
                or age > self.max_observation_age_s
            ):
                self._clear_locked("BACK_NOT_CONFIRMABLE")
                return {
                    "ok": False,
                    "code": "back_not_confirmable",
                    "message": "One current person is required before confirming back",
                    "back_confirmation": self._snapshot_locked(),
                }
            if type(preview_frame_index) is not int:
                self._clear_locked("PREVIEW_FRAME_NOT_BOUND")
                return {
                    "ok": False,
                    "code": "preview_frame_not_bound",
                    "message": "Confirm Back requires the displayed preview frame",
                    "back_confirmation": self._snapshot_locked(),
                }
            person_frame_index = int(decision.get("frame_index", -1))
            if (
                preview_frame_index < person_frame_index
                or preview_frame_index - person_frame_index > self.max_preview_lag_frames
            ):
                self._clear_locked("PREVIEW_FRAME_NOT_BOUND")
                return {
                    "ok": False,
                    "code": "preview_frame_not_bound",
                    "message": "The displayed frame is not bound to the current person result",
                    "back_confirmation": self._snapshot_locked(),
                }
            box = _finite_array(decision.get("box"), (4,))
            shape = decision.get("frame_shape")
            if box is None or not isinstance(shape, (list, tuple)) or len(shape) != 2:
                self._clear_locked("BACK_NOT_CONFIRMABLE")
                return {
                    "ok": False,
                    "code": "back_not_confirmable",
                    "message": "The current person observation is incomplete",
                    "back_confirmation": self._snapshot_locked(),
                }
            self._active = True
            # A new matching inference frame must arrive after the click before
            # Start can use this confirmation.
            self._mark_not_current_locked()
            self._validity_epoch += 1
            self._confirmed_box = [float(value) for value in box]
            self._confirmed_shape = [int(shape[0]), int(shape[1])]
            self._confirmation_frame_index = person_frame_index
            self._confirmation_preview_frame_index = preview_frame_index
            self._activation_after_frame_index = max(
                person_frame_index, preview_frame_index,
            )
            self._missed_frames = 0
            self._reason = "BACK_CONFIRMATION_PENDING_FRAME"
            return {
                "ok": True,
                "code": "back_confirmation_pending",
                "message": "Back confirmed; hold position",
                "back_confirmation": self._snapshot_locked(),
            }

    confirm_latest = confirm_back

    def begin_start_confirmation(self):
        now = float(self._clock())
        with self._lock:
            self._expire_stale_locked(now)
            if not self._active or not self._current:
                raise BackConfirmationError("Back confirmation is not active")
            token = (
                self._confirmation_frame_index,
                self._confirmation_preview_frame_index,
                self._validity_epoch,
            )
            return token, self._snapshot_locked(now=now)

    def finish_start_confirmation(self, token):
        now = float(self._clock())
        with self._lock:
            return self._validate_start_locked(token, now=now)

    def _validate_start_locked(self, token, *, now):
        self._expire_stale_locked(float(now))
        expected = (
            self._confirmation_frame_index,
            self._confirmation_preview_frame_index,
            self._validity_epoch,
        )
        if token != expected or not self._active or not self._current:
            raise BackConfirmationError("Back confirmation changed during Start")
        return self._snapshot_locked(now=now)

    def commit_start_confirmation(self, token, *, prepare, arm):
        """Persist then arm Start while excluding a concurrent gate transition."""

        if not callable(prepare) or not callable(arm):
            raise TypeError("Start confirmation callbacks must be callable")
        with self._lock:
            snapshot = self._validate_start_locked(token, now=float(self._clock()))
            prepare(snapshot)
            # A re-entrant persistence callback is allowed to surface a fault by
            # clearing the gate. Revalidate before the irreversible arm step.
            snapshot = self._validate_start_locked(token, now=float(self._clock()))
            arm(snapshot)
            return snapshot

    def assess(self, frame) -> dict:
        try:
            raw = self.person_gate.assess(frame)
        except Exception:
            self.clear_confirmation("PERSON_GATE_ERROR")
            raise
        if not isinstance(raw, dict):
            self.clear_confirmation("PERSON_GATE_ERROR")
            raise TypeError("person presence gate must return a dict")
        decision = dict(raw)
        decision["frame_index"] = int(decision.get("frame_index", frame.index))
        now = float(self._clock())
        observed_at = getattr(frame, "t", now)
        try:
            observed_at = float(observed_at)
        except (TypeError, ValueError) as exc:
            self.clear_confirmation("PERSON_FRAME_TIME_INVALID")
            raise PosteriorGateError("person frame time is invalid") from exc
        if not math.isfinite(observed_at):
            self.clear_confirmation("PERSON_FRAME_TIME_INVALID")
            raise PosteriorGateError("person frame time is invalid")
        with self._lock:
            previous_index = (
                None if self._latest_decision is None
                else int(self._latest_decision.get("frame_index", -1))
            )
            if previous_index is not None and decision["frame_index"] <= previous_index:
                if (
                    self._active
                    and not self._current
                    and self._activation_after_frame_index is not None
                    and decision["frame_index"] <= self._activation_after_frame_index
                ):
                    snapshot = self._snapshot_locked(now=now)
                    return {
                        **decision,
                        "posterior_present": False,
                        "reason": "BACK_CONFIRMATION_PENDING_FRAME",
                        "operator_confirmed": False,
                        "confirmation_active": True,
                        "confirmation_frame_index": self._confirmation_frame_index,
                        "back_confirmation": snapshot,
                    }
                self._clear_locked("NON_MONOTONIC_PERSON_FRAME")
                snapshot = self._snapshot_locked(now=now)
                return {
                    **decision,
                    "posterior_present": False,
                    "reason": "NON_MONOTONIC_PERSON_FRAME",
                    "operator_confirmed": False,
                    "confirmation_active": False,
                    "confirmation_frame_index": None,
                    "back_confirmation": snapshot,
                }
            self._latest_decision = dict(decision)
            self._latest_observed_at = observed_at
            if decision.get("person_present") is not True:
                self._mark_not_current_locked()
                self._missed_frames += 1 if self._active else 0
                raw_reason = str(decision.get("reason", "NO_PERSON"))
                if raw_reason in {"AMBIGUOUS_PEOPLE", "PERSON_DETECTION_INVALID"}:
                    self._clear_locked(raw_reason)
                elif self._active and self._missed_frames >= self.max_missed_frames:
                    self._clear_locked("CONFIRMATION_TARGET_LOST")
                else:
                    self._reason = raw_reason
                reason = self._reason
                snapshot = self._snapshot_locked()
                return {
                    **decision,
                    "posterior_present": False,
                    "reason": reason,
                    "operator_confirmed": False,
                    "confirmation_active": bool(self._active),
                    "confirmation_frame_index": self._confirmation_frame_index,
                    "back_confirmation": snapshot,
                }

            observed_box = decision.get("box")
            observed_shape = decision.get("frame_shape")
            if not self._active:
                self._mark_not_current_locked()
                self._missed_frames = 0
                self._reason = "BACK_CONFIRMATION_REQUIRED"
            elif (
                list(observed_shape or ()) != list(self._confirmed_shape or ())
                or not _boxes_match(
                    self._confirmed_box,
                    observed_box,
                    frame_shape=observed_shape,
                    min_iou=self.min_iou,
                    max_center_shift_fraction=self.max_center_shift_fraction,
                    min_area_ratio=self.min_area_ratio,
                    max_area_ratio=self.max_area_ratio,
                )
            ):
                self._clear_locked("CONFIRMED_PERSON_CHANGED")
            else:
                self._confirmed_box = [float(value) for value in observed_box]
                self._missed_frames = 0
                if decision["frame_index"] <= int(self._activation_after_frame_index):
                    self._mark_not_current_locked()
                    self._reason = "BACK_CONFIRMATION_PENDING_FRAME"
                else:
                    self._current = True
                    self._reason = "OPERATOR_CONFIRMED_BACK"
            snapshot = self._snapshot_locked()
            return {
                **decision,
                "posterior_present": bool(self._active and self._current),
                "reason": self._reason,
                "operator_confirmed": bool(self._active and self._current),
                "confirmation_active": bool(self._active),
                "confirmation_frame_index": self._confirmation_frame_index,
                "back_confirmation": snapshot,
            }


class UltralyticsPersonGate:
    """Persistent YOLO person detector evaluated on the exact landmark frame."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device=0,
        image_size: int = 640,
        model_release_id: Optional[str] = None,
        model_sha256: Optional[str] = None,
        model_factory=None,
    ) -> None:
        path = Path(checkpoint)
        if path.is_symlink() or not path.is_file():
            raise PosteriorGateError("posterior gate checkpoint must be a regular file")
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("posterior gate image_size must be a positive integer")
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except Exception as exc:  # noqa: BLE001 - becomes a fail-closed model init error
                raise PosteriorGateError("ultralytics runtime import failed") from exc
            model_factory = YOLO
        try:
            self.model = model_factory(str(path))
        except Exception as exc:  # noqa: BLE001 - no fallback model is permitted
            raise PosteriorGateError("posterior gate model load failed") from exc
        self.checkpoint = str(path)
        self.device = device
        self.image_size = image_size
        self.model_release_id = model_release_id
        self.model_sha256 = model_sha256
        self._lock = threading.Lock()

    def assess(self, frame) -> dict:
        rgb = getattr(frame, "frame", None)
        index = int(getattr(frame, "index"))
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise PosteriorGateError("posterior gate requires one RGB HxWx3 frame")
        # Ultralytics' ndarray source contract is OpenCV BGR. CameraStream's
        # public frame contract is RGB, so convert explicitly and contiguously.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        started = time.monotonic()
        try:
            with self._lock:
                batch = self.model.predict(
                    bgr,
                    device=self.device,
                    conf=0.10,
                    imgsz=self.image_size,
                    verbose=False,
                    classes=[0],
                )
        except Exception as exc:  # noqa: BLE001 - fail closed; never synthesize presence
            raise PosteriorGateError("posterior gate inference failed") from exc
        if not batch:
            raise PosteriorGateError("posterior gate returned no result object")
        result = batch[0]
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            try:
                raw_boxes = boxes.xyxy.detach().cpu().numpy()
                raw_box_confidence = boxes.conf.detach().cpu().numpy()
            except Exception as exc:  # noqa: BLE001
                raise PosteriorGateError("posterior gate output schema is invalid") from exc
            if len(raw_boxes) != len(raw_box_confidence):
                raise PosteriorGateError("posterior gate output lengths disagree")
            for box, box_confidence in zip(raw_boxes, raw_box_confidence):
                detections.append({
                    "box": box,
                    "box_confidence": box_confidence,
                })
        decision = select_single_person(detections, image_shape=rgb.shape[:2])
        decision.update({
            "frame_index": index,
            "inference_ms": (time.monotonic() - started) * 1000.0,
            "model_release_id": self.model_release_id,
            "model_sha256": self.model_sha256,
        })
        return decision
