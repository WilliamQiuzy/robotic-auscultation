"""Web console for point-conditioned auscultation data collection.

One page the doctor + operator watch: the live camera with the current auscultation target overlaid,
a feedback HUD (state / force / audio / point list), and keyboard controls. The backend
(this server, on linbox) owns the camera + sweep state; the browser is just display + controls, so the
UI is decoupled from the hardware.

Keys:  SPACE = confirm (heard a good sound → next point)   R = redo (keep trying at this point)
       ← / → = previous / next point                       S = start / stop recording   F = fullscreen

  python scripts/collect_server.py --fake                 # synthetic camera + sim sensors (any machine)
  python scripts/collect_server.py --camera-backend orbbec --out ~/wam_data/sessions   # real (linbox)

--fake drives the REAL PointSweep/overlay with a simulated scene + sensors, so the whole console works
without hardware; on linbox use the Orbbec camera, live force/audio, robot telemetry, and teleop.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import signal
import stat
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

_ROBOT_SUMMARY_SCHEMA_V3 = "wam_openarm_exact_control_capture_summary_v3"
_ROBOT_SUMMARY_SCHEMA_V4 = "wam_openarm_exact_control_capture_summary_v4"
_ROBOT_ENVELOPE_SCHEMA_V4 = "wam_openarm_control_envelope_v4"
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


def _finite_json(o):
    """Recursively replace non-finite floats (Infinity/NaN) with None so the served
    /state is strict JSON the browser can parse.

    Python's json emits bare ``Infinity``/``NaN`` tokens (invalid JSON); e.g. a
    ``tracking_age_s`` of ``inf`` before the first tracking result would break
    ``JSON.parse`` in the browser and blank the console.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite_json(v) for v in o]
    return o


def _robot_v3_tx_summary_valid(summary):
    """Return whether diagnostic v3 evidence proves all 7+1 TX writes per tick."""

    if not isinstance(summary, Mapping):
        return False
    tick_count = summary.get("control_tick_count")
    count_fields = (
        "arm_tx_expected_frame_count",
        "arm_tx_full_frame_write_count",
        "arm_tx_failed_frame_write_count",
        "gripper_tx_expected_frame_count",
        "gripper_tx_full_frame_write_count",
        "gripper_tx_failed_frame_write_count",
        "socketcan_tx_enqueue_failure_tick_count",
    )
    return (
        summary.get("schema") == _ROBOT_SUMMARY_SCHEMA_V3
        and summary.get("status")
        == "exact_control_capture_validated_socketcan_tx_enqueue"
        and summary.get("send_status_observable") is True
        and summary.get("send_status_reason") == _ROBOT_TX_STATUS_REASON
        and summary.get("send_status_semantics") == _ROBOT_TX_STATUS_SEMANTICS
        and summary.get("motor_ack_observable") is False
        and summary.get("motor_ack_reason") == _ROBOT_MOTOR_ACK_REASON
        and summary.get("feedback_liveness_observable") is False
        and summary.get("feedback_liveness_reason")
        == _ROBOT_FEEDBACK_LIVENESS_REASON
        and summary.get("training_eligible") is False
        and summary.get("eligibility_blockers")
        == ["robot_feedback_liveness_unobservable"]
        and type(tick_count) is int
        and tick_count > 0
        and all(type(summary.get(field)) is int for field in count_fields)
        and summary["arm_tx_expected_frame_count"] == tick_count * 7
        and summary["arm_tx_full_frame_write_count"] == tick_count * 7
        and summary["arm_tx_failed_frame_write_count"] == 0
        and summary["gripper_tx_expected_frame_count"] == tick_count
        and summary["gripper_tx_full_frame_write_count"] == tick_count
        and summary["gripper_tx_failed_frame_write_count"] == 0
        and summary["socketcan_tx_enqueue_failure_tick_count"] == 0
    )


def _robot_v4_summary_valid(summary, *, contract=None):
    """Return whether a v4 End receipt is clean and training-eligible."""

    if not isinstance(summary, Mapping):
        return False
    tick_count = summary.get("control_tick_count")
    count_fields = (
        "arm_tx_expected_frame_count",
        "arm_tx_full_frame_write_count",
        "arm_tx_failed_frame_write_count",
        "gripper_tx_expected_frame_count",
        "gripper_tx_full_frame_write_count",
        "gripper_tx_failed_frame_write_count",
        "socketcan_tx_enqueue_failure_tick_count",
    )
    policy_fields = (
        "buffer_duration_ns",
        "max_packets",
        "max_gap_ns",
        "max_staleness_ns",
        "max_exposure_interval_ns",
    )
    if contract is not None and any(
        summary.get(name) != getattr(contract, name, None)
        for name in policy_fields
    ):
        return False
    return (
        summary.get("schema") == _ROBOT_SUMMARY_SCHEMA_V4
        and summary.get("status")
        == "exact_control_capture_validated_feedback_liveness"
        and summary.get("send_status_observable") is True
        and summary.get("send_status_reason") == _ROBOT_TX_STATUS_REASON
        and summary.get("send_status_semantics") == _ROBOT_TX_STATUS_SEMANTICS
        and summary.get("motor_ack_observable") is False
        and summary.get("motor_ack_reason") == _ROBOT_MOTOR_ACK_REASON
        and summary.get("feedback_liveness_observable") is True
        and summary.get("feedback_liveness_reason")
        == "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
        and summary.get("feedback_liveness_semantics")
        == "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
        and summary.get("feedback_counter_progress_all_motors") is True
        and summary.get("feedback_all_ticks_fresh") is True
        and summary.get("feedback_gap_clean") is True
        and summary.get("feedback_clean") is True
        and summary.get("feedback_gap_violation_count_delta") == [0] * 8
        and summary.get("trajectory_validation_schema")
        == "wam_openarm_trajectory_validation_v1"
        and summary.get("trajectory_nondegenerate") is True
        and summary.get("command_response_consistent") is True
        and summary.get("command_response_violation_joint_names") == []
        and summary.get("training_eligible") is True
        and summary.get("eligibility_blockers") == []
        and type(tick_count) is int
        and tick_count > 0
        and all(type(summary.get(field)) is int for field in count_fields)
        and all(
            type(summary.get(field)) is int and summary[field] > 0
            for field in policy_fields
        )
        and summary["arm_tx_expected_frame_count"] == tick_count * 7
        and summary["arm_tx_full_frame_write_count"] == tick_count * 7
        and summary["arm_tx_failed_frame_write_count"] == 0
        and summary["gripper_tx_expected_frame_count"] == tick_count
        and summary["gripper_tx_full_frame_write_count"] == tick_count
        and summary["gripper_tx_failed_frame_write_count"] == 0
        and summary["socketcan_tx_enqueue_failure_tick_count"] == 0
    )

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path[:0] = [ROOT, os.path.join(ROOT, "ausc", "perception"), os.path.join(ROOT, "ausc")]
from ausc_targets import load_targets                               # noqa: E402
from collection_platform import (DIAGNOSTIC_PERCEPTION_CONTRACT,      # noqa: E402
                                 PREVIEW_ONLY_CONTRACT,
                                 PRODUCTION_TELEOP_CONTRACT,
                                 CollectionPlatform, CollectionState,
                                 reconcile_orphan_diagnostic_sessions)
from perception_capture import (DiagnosticPerceptionCapture,        # noqa: E402
                                ProductionTeleopCapture,
                                recover_orphan_diagnostic_video)
from perception_worker import (BoundedFifoWorker, LatestFrameWorker,  # noqa: E402
                               TrackerProcessor, verify_file_sha256)
from posterior_gate import (BackConfirmationError,                  # noqa: E402
                            OperatorConfirmedBackGate,
                            UltralyticsPersonGate)
from health import AUDIO_FLOOR_DB                                   # noqa: E402
from ausc.perception.robot.frame_evidence_bundle import (          # noqa: E402
    FrameEvidenceBundle,
)
from ausc.perception.robot.robot_capture_coordinator import (       # noqa: E402
    RobotCaptureCoordinatorError,
    RobotCaptureNotReady,
)
from ausc.perception.robot.trusted_controller_companion import (    # noqa: E402
    CompanionError,
    CompanionFault,
    TrustedControllerCompanion,
)
from ausc.perception.production_runtime_activation import (        # noqa: E402
    ProductionRuntimeActivationError,
    load_completed_handeye_release,
    load_production_runtime_assembly,
)
from ausc.perception import depth_codec                            # noqa: E402


def validate_audio_hardware_descriptor(
    observed,
    *,
    expected_vid,
    expected_pid,
    expected_serial,
    expected_port_path,
    channels,
    pick_channel,
):
    """Fail closed unless the resolved PortAudio input is the commissioned USB path."""

    if not isinstance(observed, Mapping):
        raise RuntimeError("audio hardware descriptor is unavailable")
    descriptor = dict(observed)
    if str(descriptor.get("hostapi_name", "")).casefold() != "alsa":
        raise RuntimeError("audio host API must be ALSA")
    portaudio_index = descriptor.get("portaudio_index")
    max_inputs = descriptor.get("max_input_channels")
    if type(portaudio_index) is not int or portaudio_index < 0:
        raise RuntimeError("audio PortAudio index is invalid")
    if type(max_inputs) is not int or max_inputs <= 0:
        raise RuntimeError("audio input channel capacity is invalid")
    if type(channels) is not int or channels <= 0 or channels > max_inputs:
        raise RuntimeError("audio opened channel count exceeds the resolved device")
    if type(pick_channel) is not int or not 0 <= pick_channel < channels:
        raise RuntimeError("audio pick channel is invalid")
    if descriptor.get("usb_vid") != expected_vid or descriptor.get("usb_pid") != expected_pid:
        raise RuntimeError("audio USB VID/PID does not match the commissioned device")
    if expected_serial is not None:
        if descriptor.get("usb_serial") != expected_serial:
            raise RuntimeError("audio USB serial does not match the commissioned device")
    else:
        if not expected_port_path or descriptor.get("usb_port_path") != expected_port_path:
            raise RuntimeError("audio USB topology does not match the commissioned device")
    return descriptor


class CollectionHTTPServer(ThreadingHTTPServer):
    """HTTP server whose long-lived MJPEG clients cannot block shutdown."""

    daemon_threads = True
    block_on_close = False


def _verified_regular_file(path):
    """Return (sha256, size, identity) without following a replaced symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"completion artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"completion artifact changed while hashing: {path}")
        return digest.hexdigest(), int(after.st_size), identity_after
    finally:
        os.close(descriptor)


def _verified_immutable_input(value, *, label):
    """Reject symlinks/dynamic pointers before hashing a pinned config file."""

    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute immutable path")
    dynamic = {"current", "latest", "active", ".wam-current", "wam-current"}
    if any(part.lower() in dynamic for part in path.parts):
        raise RuntimeError(f"{label} cannot use a dynamic pointer")
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing") from exc
    if not stat.S_ISREG(info.st_mode) or resolved != path:
        raise RuntimeError(
            f"{label} must be a real immutable non-symlink file"
        )
    digest, size, identity = _verified_regular_file(path)
    return path, digest, size, identity


def _regular_file_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"completion artifact is not a regular file: {path}")
    return (
        info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def visible_tracking_points(
    tracking, *, max_coast_std_m=0.01, expected_names=None,
):
    """Return only model-backed points that are safe to display.

    Real mode must never fall back to ``DEFAULT_TARGETS``: an absent back, an
    initializing/reacquiring tracker, or a globally lost track yields no points.
    Legacy short coasts remain visually distinct. Production robot occlusion is
    stricter: only a current DIRECT point or a low-uncertainty
    PATIENT_FRAME_LOCKED point from the independent patient pose may display.
    """
    if not isinstance(tracking, dict):
        return []
    if tracking.get("back_present") is not True or tracking.get("initialized") is not True:
        return []
    if str(tracking.get("reason", "TRACKING")) != "TRACKING":
        return []
    visible = []
    for raw in tracking.get("points") or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", "LOST"))
        occlusion_state = str(raw.get("occlusion_state", status))
        measured = raw.get("measured") is True
        try:
            pixel = [float(raw["pixel"][0]), float(raw["pixel"][1])]
            position_std = float(raw.get("position_std", float("inf")))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(value) for value in pixel) or not math.isfinite(position_std):
            continue
        if measured and status == "TRACKING":
            display_status = "measured"
        elif (
            measured
            and status == "DIRECT"
            and occlusion_state == "DIRECT"
            and raw.get("valid") is True
            and raw.get("robot_occluded") is False
            and raw.get("direct_geometry_accepted") is True
            and raw.get("source") == "current_direct_geometry"
        ):
            display_status = "measured"
        elif (
            not measured
            and status == "PATIENT_FRAME_LOCKED"
            and occlusion_state == "PATIENT_FRAME_LOCKED"
            and raw.get("valid") is True
            and raw.get("robot_occluded") is True
            and raw.get("source") == "independent_patient_pose"
            and position_std <= max_coast_std_m
        ):
            display_status = "patient_locked"
        elif status == "COASTING" and position_std <= max_coast_std_m:
            display_status = "coasting"
        else:
            continue
        point = copy.deepcopy(raw)
        point["pixel"] = pixel
        point["position_std"] = position_std
        point["display_status"] = display_status
        visible.append(point)
    if expected_names is None:
        return visible
    expected = tuple(str(name) for name in expected_names)
    by_name = {}
    for point in visible:
        name = str(point.get("name", ""))
        if name in by_name:
            return []
        by_name[name] = point
    if set(by_name) != set(expected) or len(by_name) != len(expected):
        return []
    return [by_name[name] for name in expected]


def draw_tracking_overlay(
    frame, tracking, *, active_name, completed_names=(), expected_names=None,
):
    """Draw only the current tracker output; a no-back result leaves RGB untouched."""
    import cv2

    completed = set(completed_names)
    for point in visible_tracking_points(tracking, expected_names=expected_names):
        name = str(point.get("name", ""))
        px, py = (int(round(value)) for value in point["pixel"])
        if name in completed:
            color, radius, thickness = (0, 200, 0), 9, 2
            label = f"{name} OK"
        elif name == active_name:
            color, radius, thickness = (0, 220, 255), 22, 3
            label = f"GO HERE: {name}"
        else:
            color, radius, thickness = (150, 150, 150), 8, 2
            label = name
        if point["display_status"] in {"coasting", "patient_locked"}:
            # Dashed circle: the point is a short, explicitly uncertain coast.
            for segment in range(0, 16, 2):
                cv2.ellipse(
                    frame, (px, py), (radius, radius), 0,
                    segment * 360 / 16, (segment + 1) * 360 / 16, color, thickness,
                )
            label += (
                " (locked)"
                if point["display_status"] == "patient_locked"
                else " (coasting)"
            )
        else:
            cv2.circle(frame, (px, py), radius, color, thickness)
        cv2.circle(frame, (px, py), 3, color, -1)
        cv2.putText(
            frame, label, (px - 50, py - radius - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1,
        )
    return frame


def effective_tracking(
    tracking, *, expected_names, now, max_age_s, current_depth_plausible,
):
    """Return the only constellation allowed to drive the live UI controls.

    The model result must be recent, refer to a currently plausible RGB-D scene,
    and contain each expected landmark exactly once.  Any failure hides all points
    instead of retaining stale or partial anatomy.
    """
    result = copy.deepcopy(tracking) if isinstance(tracking, dict) else {}
    result.setdefault("back_present", False)
    result.setdefault("initialized", False)
    result.setdefault("reason", "MODEL_STARTING")
    result["constellation_valid"] = False
    result["points"] = []

    if current_depth_plausible is not True:
        result.update(
            back_present=False,
            initialized=False,
            reason="NO_BACK_CURRENT_FRAME",
        )
        return result

    source_t = tracking.get("camera_host_t_s", tracking.get("host_t")) \
        if isinstance(tracking, dict) else None
    try:
        age_s = max(0.0, float(now) - float(source_t))
    except (TypeError, ValueError):
        age_s = float("inf")
    result["tracking_age_s"] = age_s
    if not math.isfinite(age_s) or age_s > float(max_age_s):
        result.update(
            back_present=False,
            initialized=False,
            reason="TRACKING_STALE",
        )
        return result

    if tracking.get("back_present") is not True:
        result.update(back_present=False, initialized=False)
        return result
    if tracking.get("initialized") is not True:
        result["initialized"] = False
        return result

    points = visible_tracking_points(tracking, expected_names=expected_names)
    if len(points) != len(tuple(expected_names)):
        result["reason"] = "INCOMPLETE_CONSTELLATION"
        return result
    result["points"] = points
    result["constellation_valid"] = True
    return result


def tracking_for_preview_frame(frame_index, tracking):
    """Never paint a positive inference result onto another camera frame."""

    result = copy.deepcopy(tracking) if isinstance(tracking, dict) else {}
    try:
        same_frame = int(result.get("frame_index", -1)) == int(frame_index)
    except (TypeError, ValueError):
        same_frame = False
    if same_frame:
        return result
    return {
        "frame_index": int(frame_index),
        "back_present": False,
        "initialized": False,
        "constellation_valid": False,
        "detector_confidence": 0.0,
        "reason": "MODEL_PENDING_FOR_CURRENT_FRAME",
        "points": [],
    }


def perception_stage_snapshot(
    *, model_status, tracking, back_confirmation, model_release_id,
):
    """Describe the person gate and V9 landmark stage independently.

    The shared worker is healthy while it is running the person gate, even
    though the gated V9 tracker has not yet executed.  Keep those states
    separate so ``RUNNING`` can no longer be mistaken for ten-point inference.
    """

    current = tracking if isinstance(tracking, dict) else {}
    gate = current.get("posterior_gate")
    gate = gate if isinstance(gate, dict) else {}
    confirmation = (
        back_confirmation if isinstance(back_confirmation, dict) else {}
    )
    probe = current.get("landmark_probe")
    probe = probe if isinstance(probe, dict) else {}
    probe_active = probe.get("executed") is True
    worker_status = str(model_status or "STARTING")
    if worker_status in {"ERROR", "BLOCKED"}:
        gate_status = "ERROR"
    elif worker_status in {"STARTING", "RECONNECTING", "WARMING_UP"}:
        gate_status = "STARTING"
    else:
        gate_status = "RUNNING"

    confirmed = bool(
        confirmation.get("confirmed") is True
        or gate.get("operator_confirmed") is True
    )
    reason = str(current.get("reason", "MODEL_STARTING"))
    initialized = current.get("initialized") is True
    if gate_status == "ERROR":
        landmark_status = "ERROR"
    elif gate_status == "STARTING":
        landmark_status = "LOADING"
    elif not confirmed:
        landmark_status = "PROBING" if probe_active else "LOADED_WAITING_FOR_BACK"
    elif reason == "TRACKING" and initialized:
        landmark_status = "TRACKING"
    elif reason == "REACQUIRING":
        landmark_status = "REACQUIRING"
    else:
        landmark_status = "INITIALIZING"
    publication_authorized = bool(confirmed and gate_status == "RUNNING")
    inference_active = bool(
        gate_status == "RUNNING"
        and (
            probe_active
            or (
                publication_authorized
                and landmark_status in {
                    "INITIALIZING", "TRACKING", "REACQUIRING",
                }
            )
        )
    )
    try:
        frame_index = int(
            probe.get("frame_index") if probe_active
            else current.get("frame_index")
        )
    except (TypeError, ValueError):
        frame_index = None
    if gate_status == "ERROR":
        landmark_mode = "ERROR"
    elif gate_status == "STARTING":
        landmark_mode = "LOADING"
    elif probe_active and not publication_authorized:
        landmark_mode = "GATED_PROBE"
    elif publication_authorized:
        landmark_mode = "CONFIRMED_TRACKING"
    else:
        landmark_mode = "WAITING_FOR_BACK"
    return {
        "person_gate": {
            "status": gate_status,
            "reason": str(gate.get("reason", reason)),
        },
        "landmark_model": {
            "status": landmark_status,
            "mode": landmark_mode,
            "inference_active": bool(inference_active),
            "publication_authorized": publication_authorized,
            "last_result_frame_index": frame_index if inference_active else None,
            "model_release_id": str(model_release_id or "unavailable"),
        },
    }


def validate_capture_boundary_closure(
    boundary, *, n_rgb_frames, expected_fps, final_frame_range=None,
):
    """Prove Start/End, camera FIFO, disk FIFO, and final indexes are one interval."""

    if not isinstance(boundary, dict):
        raise RuntimeError("capture boundary closure proof is missing")
    try:
        count = int(n_rgb_frames)
        fps = float(expected_fps)
        start_index = int(boundary["camera_start_boundary_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("capture boundary closure metadata is incomplete") from exc
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("capture boundary count/FPS is invalid")
    expected_first = start_index + 1
    expected_last = expected_first + count - 1
    end = boundary.get("camera_end_boundary_proof")
    camera = boundary.get("camera_delivery_fifo")
    ingress = boundary.get("capture_ingress_fifo")
    if not all(isinstance(item, dict) for item in (end, camera, ingress)):
        raise RuntimeError("capture boundary closure proof is incomplete")
    if (
        end.get("boundary_latched_atomically") is not True
        or end.get("post_boundary_overwrite_prevented") is not True
    ):
        raise RuntimeError("camera boundary closure was not atomic")
    if (
        end.get("thread_alive_at_boundary") is not True
        or end.get("thread_error_at_boundary") is not None
    ):
        raise RuntimeError("camera thread was not healthy at the End boundary")
    try:
        snapshot_index = int(end["snapshot_frame_index"])
        snapshot_age = float(end["snapshot_age_s"])
        proof_limit = float(end["max_snapshot_age_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("camera End snapshot freshness proof is incomplete") from exc
    allowed_age = min(3.0 / fps, proof_limit)
    if (
        not math.isfinite(snapshot_age)
        or snapshot_age < 0
        or not math.isfinite(allowed_age)
        or allowed_age <= 0
        or snapshot_age > allowed_age
    ):
        raise RuntimeError("camera End snapshot is stale")
    camera_ok = (
        camera.get("status") == "OK"
        and camera.get("thread_alive") is True
        and camera.get("frame_fresh") is True
        and camera.get("interval_active") is False
        and camera.get("accepting") is False
        and int(camera.get("queue_depth", -1)) == 0
        and int(camera.get("interval_overflow_frames", -1)) == 0
    )
    if not camera_ok:
        raise RuntimeError("camera FIFO closure health is invalid")
    try:
        camera_enqueued = int(camera["interval_enqueued_frames"])
        camera_delivered = int(camera["interval_delivered_frames"])
        camera_first = int(camera["first_enqueued_frame_index"])
        camera_last = int(camera["last_enqueued_frame_index"])
        camera_delivered_last = int(camera["last_delivered_frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("camera FIFO closure indexes are incomplete") from exc
    if camera_enqueued != count or camera_delivered != count:
        raise RuntimeError("camera FIFO frame count does not close on final capture")
    if (
        camera_first != expected_first
        or camera_last != expected_last
        or camera_delivered_last != expected_last
        or snapshot_index != expected_last
    ):
        raise RuntimeError("camera FIFO/index closure does not match the End snapshot")
    ingress_ok = (
        ingress.get("status") == "OK"
        and ingress.get("thread_alive") is True
        and ingress.get("interval_active") is False
        and int(ingress.get("queue_depth", -1)) == 0
        and int(ingress.get("overflow_rejections", -1)) == 0
        and int(ingress.get("abandoned_after_error", -1)) == 0
    )
    if not ingress_ok:
        raise RuntimeError("capture ingress closure health is invalid")
    try:
        accepted = int(ingress["accepted"])
        processed = int(ingress["processed"])
        ingress_first = int(ingress["first_accepted_index"])
        ingress_last = int(ingress["last_accepted_index"])
        completed_last = int(ingress["last_completed_index"])
        submitted_last = int(boundary["last_submitted_camera_index"])
        boundary_completed_last = int(boundary["last_completed_camera_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("capture ingress closure indexes are incomplete") from exc
    if accepted != count or processed != count:
        raise RuntimeError("capture ingress frame count does not close on final capture")
    if (
        ingress_first != expected_first
        or ingress_last != expected_last
        or completed_last != expected_last
        or submitted_last != expected_last
        or boundary_completed_last != expected_last
    ):
        raise RuntimeError("capture ingress/index closure does not match the End snapshot")
    if final_frame_range is not None:
        try:
            final_first = int(final_frame_range["first_frame_index"])
            final_last = int(final_frame_range["last_frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("final capture index range is incomplete") from exc
        if final_first != expected_first or final_last != expected_last:
            raise RuntimeError("final artifact index closure does not match the End snapshot")
    return {
        "closed": True,
        "frame_count": count,
        "first_frame_index": expected_first,
        "last_frame_index": expected_last,
        "snapshot_age_s": snapshot_age,
        "snapshot_age_limit_s": allowed_age,
    }


def open_camera(camera_factory, *, backend, index, camera_id=None, **camera_options):
    """Open the requested camera or raise; production must never synthesize data."""
    camera = camera_factory(
        backend=backend, index=index, camera_id=camera_id, rgb=True,
        **camera_options,
    )
    return camera.start()


def tracking_payload(frame, tracking, *, model_release_id, model_sha256):
    """Bind one tracker decision to the exact camera clocks and model asset."""
    payload = copy.deepcopy(tracking) if isinstance(tracking, dict) else {}
    payload.update({
        "frame_index": int(frame.index),
        "camera_host_t_s": float(frame.t),
        "camera_device_t_ms": (
            None if getattr(frame, "device_t_ms", None) is None
            else float(frame.device_t_ms)
        ),
        "camera_age_s": float(getattr(frame, "age_s", 0.0)),
        "camera_status": str(getattr(frame, "status", "NO_DATA")),
        "model_release_id": str(model_release_id),
        "model_sha256": str(model_sha256),
    })
    return payload


def associate_tracking_to_camera_frame(frame, tracking):
    """Bind the latest completed inference to one camera-rate capture frame.

    The outer ``frame_index`` always names the captured camera frame. The source
    inference clock remains explicit, so carried-forward results can never be
    mistaken for same-frame inference during downstream training conversion.
    """
    camera_index = int(frame.index)
    camera_host_t = float(frame.t)
    if not math.isfinite(camera_host_t):
        raise RuntimeError(f"camera frame {camera_index} has a non-finite timestamp")
    source = copy.deepcopy(tracking) if isinstance(tracking, dict) else {}
    try:
        source_index = int(source.get("frame_index", 0))
    except (TypeError, ValueError):
        source_index = 0
    if source_index <= 0:
        return {
            "frame_index": camera_index,
            "host_t": camera_host_t,
            "device_t": getattr(frame, "device_t_ms", None),
            "source_tracking_frame_index": None,
            "source_tracking_host_t_s": None,
            "source_tracking_device_t_ms": None,
            "tracking_association": "unavailable",
            "tracking_age_s": None,
            "back_present": False,
            "initialized": False,
            "detector_confidence": 0.0,
            "reason": "MODEL_UNAVAILABLE_FOR_CAMERA_FRAME",
            "points": [],
        }
    if source_index > camera_index:
        raise RuntimeError(
            f"future tracking frame {source_index} cannot annotate camera frame {camera_index}"
        )
    source_host_t = source.get("camera_host_t_s", source.get("host_t"))
    try:
        source_host_t = float(source_host_t)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"tracking source frame {source_index} has no finite host timestamp"
        ) from exc
    age_s = camera_host_t - source_host_t
    if not math.isfinite(age_s) or age_s < -1e-6:
        raise RuntimeError(
            f"tracking source frame {source_index} is from the future in host time"
        )
    age_s = max(0.0, age_s)
    source_device_t = source.get("camera_device_t_ms", source.get("device_t"))
    if source_device_t is not None:
        try:
            source_device_t = float(source_device_t)
        except (TypeError, ValueError):
            source_device_t = None
    source.update({
        "frame_index": camera_index,
        "host_t": camera_host_t,
        "device_t": getattr(frame, "device_t_ms", None),
        "source_tracking_frame_index": source_index,
        "source_tracking_host_t_s": source_host_t,
        "source_tracking_device_t_ms": source_device_t,
        "tracking_association": (
            "same_camera_frame" if source_index == camera_index else "carried_forward"
        ),
        "tracking_age_s": age_s,
    })
    return source


def depth_back_plausible(depth, depth_scale=1.0):
    """Cheap current-frame fail-safe used to hide a stale overlay immediately."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or not depth.size:
        return False
    # The production detector still consumes full-resolution depth.  This is
    # only a fast current-frame kill switch, so sample the fixed grid instead
    # of sorting two million pixels on every 30 Hz frame.
    stride = max(1, min(depth.shape) // 128)
    sample = depth[::stride, ::stride]
    values_m = sample.astype(np.float32, copy=False) * (float(depth_scale) / 1000.0)
    values = values_m[np.isfinite(values_m) & (values_m > 0.2) & (values_m < 3.0)]
    if values.size < 0.12 * sample.size:
        return False
    median = float(np.median(values))
    extent = float(np.percentile(values, 90) - np.percentile(values, 10))
    return 0.4 < median < 1.6 and extent > 0.08


def clone_camera_frame(frame):
    """Detach SDK buffers before a frame crosses into GPU/disk worker threads."""
    from ausc.perception.camera_stream import CameraFrame

    return CameraFrame(
        frame=(None if frame.frame is None else np.ascontiguousarray(frame.frame.copy())),
        depth=(None if frame.depth is None else np.ascontiguousarray(frame.depth.copy())),
        depth_scale=float(frame.depth_scale),
        t=float(frame.t),
        index=int(frame.index),
        status=str(frame.status),
        age_s=float(getattr(frame, "age_s", 0.0)),
        device_t_ms=getattr(frame, "device_t_ms", None),
        time_source=str(getattr(frame, "time_source", "host_arrival")),
    )


def placeholder_frame(text):
    """Render an honest camera/model failure frame; never synthesize anatomy."""
    import cv2

    image = np.full((480, 640, 3), (28, 20, 14), np.uint8)
    cv2.putText(
        image, str(text), (24, 240), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (150, 170, 200), 1,
    )
    return image


def encode_jpeg(bgr):
    import cv2

    ok, buffer = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not ok:
        raise RuntimeError("preview JPEG encoding failed")
    return buffer.tobytes()


DEFAULT_PRODUCTION_BLOCKERS = (
    "physical audio-force sync calibration is missing",
    "Orbbec mode-bound hardware LED latency calibration is missing",
    "complete robot-mask and robot-base runtime activation is missing",
    "confirmed direct motor-disable command is missing",
    "live robot/action, Hall force, and stethoscope audio identities are not commissioned",
)


def _resolve_production_blockers(cli_blockers):
    """Resolve the effective commissioning blocker set from the CLI.

    The built-in fail-closed blockers are IMMUTABLE from the command line: an
    unauthenticated string must never be able to attest commissioning complete.  Any
    ``--production-blocker`` values are APPENDED (to declare additional known-bad
    conditions); they can never clear or shrink the default set.  Clearing the
    commissioning-complete set is derived solely from SHA-bound activation evidence
    (the go-live activation path), never asserted by a bare CLI token.
    """
    extra = tuple(cli_blockers) if cli_blockers else ()
    return DEFAULT_PRODUCTION_BLOCKERS + extra


class Collector:
    """Single-camera preview, v9 inference, and diagnostic-only capture owner."""

    def __init__(
        self,
        targets,
        fake=True,
        camera_backend="orbbec",
        camera_index=0,
        camera_id=None,
        camera_width=None,
        camera_height=None,
        camera_fps=None,
        camera_color_format=None,
        depth_width=None,
        depth_height=None,
        depth_fps=None,
        depth_format=None,
        out_dir=None,
        *,
        checkpoint=None,
        checkpoint_sha256=None,
        base_model=None,
        base_model_sha256=None,
        model_release_id=None,
        posterior_gate_checkpoint=None,
        posterior_gate_checkpoint_sha256=None,
        posterior_gate_release_id=None,
        runtime_release_id=None,
        runtime_lock=None,
        runtime_lock_sha256=None,
        code_release_id=None,
        ffmpeg_path=None,
        ffmpeg_sha256=None,
        diagnostic_fps=20.0,
        diagnostic_min_free_bytes=5 * 1024 ** 3,
        planned_session_duration_s=60.0,
        capture_rate_margin=1.3,
        depth_compression_ratio=1.0,
        non_depth_capture_bytes_per_s=12_000_000,
        max_tracking_age_s=0.25,
        capture_queue_frames=8,
        camera_delivery_frames=32,
        camera_reopen_interval_s=2.0,
        audio_device=None,
        audio_sample_rate=44_100,
        audio_channels=2,
        audio_pick_channel=None,
        audio_expected_usb_vid=None,
        audio_expected_usb_pid=None,
        audio_expected_usb_serial=None,
        audio_expected_usb_port_path=None,
        require_audio=False,
        hall_calibration=None,
        hall_calibration_sha256=None,
        hall_port=None,
        hall_expected_usb_vid=None,
        hall_expected_usb_pid=None,
        hall_expected_usb_serial=None,
        sync_calibration=None,
        sync_calibration_sha256=None,
        require_force=None,
        require_robot=None,
        robot_capture_coordinator=None,
        controller_launcher=None,
        controller_disable=None,
        controller_preflight=None,
        controller_attest=None,
        controller_verify_motor_disable=None,
        controller_readiness_timeout_s=15.0,
        robot_mask_provider=None,
        handeye_activation=None,
        frame_evidence_bundle_factory=None,
        back_mask_provider=None,
        patient_pose_provider=None,
        occlusion_runtime=None,
        robot_camera_arm_timeout_s=2.0,
        production_blockers=DEFAULT_PRODUCTION_BLOCKERS,
        commissioning_preview_only=False,
        commissioning_hall_monitor_only=False,
        production_teleop=False,
        camera_factory=None,
        processor_factory=None,
        audio_factory=None,
        audio_resolver=None,
        force_factory=None,
        hall_resolver=None,
        sync_mode_builder=None,
        sync_loader=None,
        start_thread=True,
    ):
        self.targets = list(targets)
        self.point_names = tuple(target.name for target in self.targets)
        self.fake = bool(fake)
        self.commissioning_preview_only = bool(commissioning_preview_only)
        self.commissioning_hall_monitor_only = bool(
            commissioning_hall_monitor_only
        )
        # #6: production teleoperation capture (promotable to training). Requires a real
        # robot; mutually exclusive with preview/fake. When set, the console builds the
        # production contract + ProductionTeleopCapture writer instead of the diagnostic
        # pair. Activation is still gated by start_authorization + the finalizer.
        self.production_teleop = bool(production_teleop)
        if self.production_teleop and (self.commissioning_preview_only or self.fake):
            raise ValueError(
                "production teleop capture requires real hardware and a live robot; "
                "it is incompatible with preview-only or fake data"
            )
        if self.commissioning_preview_only and self.fake:
            raise ValueError(
                "commissioning preview requires real hardware; fake data is forbidden"
            )
        if (
            self.commissioning_hall_monitor_only
            and not self.commissioning_preview_only
        ):
            raise ValueError(
                "unsynchronized Hall monitoring is commissioning-preview-only"
            )
        self.camera_backend = "fake" if self.fake else str(camera_backend)
        self.camera_index = int(camera_index)
        self.camera_id = None if self.fake else camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.camera_color_format = camera_color_format
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.depth_format = depth_format
        self.out_dir = str(Path(out_dir or os.path.join(ROOT, "data", "sessions")).expanduser())
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.base_model = base_model
        self.base_model_sha256 = base_model_sha256
        self.model_release_id = model_release_id or ("synthetic-model" if self.fake else None)
        self.posterior_gate_checkpoint = posterior_gate_checkpoint
        self.posterior_gate_checkpoint_sha256 = posterior_gate_checkpoint_sha256
        self.posterior_gate_release_id = (
            posterior_gate_release_id or ("synthetic-posterior-gate" if self.fake else None)
        )
        self.runtime_release_id = runtime_release_id or ("synthetic-runtime" if self.fake else None)
        self.runtime_lock = runtime_lock
        self.runtime_lock_sha256 = runtime_lock_sha256
        self.code_release_id = code_release_id or ("working-tree" if self.fake else None)
        # Synthetic/demo collection must remain directly runnable with
        # ``--fake``.  It still uses the exact same concat-copy path as real
        # collection, but pins the local executable at process construction so
        # the session provenance and completion gate are deterministic.  Real
        # collection never auto-discovers a tool: main() and the configuration
        # gate require the deployment-pinned path and digest explicitly.
        if self.fake and ffmpeg_path is None and ffmpeg_sha256 is None:
            discovered_ffmpeg = shutil.which("ffmpeg")
            if discovered_ffmpeg:
                resolved_ffmpeg = Path(discovered_ffmpeg).expanduser().resolve()
                ffmpeg_path = str(resolved_ffmpeg)
                ffmpeg_sha256, _, _ = _verified_regular_file(resolved_ffmpeg)
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_sha256 = ffmpeg_sha256
        self.diagnostic_fps = float(diagnostic_fps)
        self.diagnostic_min_free_bytes = int(diagnostic_min_free_bytes)
        self.planned_session_duration_s = float(planned_session_duration_s)
        self.capture_rate_margin = max(1.0, float(capture_rate_margin))
        self.depth_compression_ratio = max(1.0, float(depth_compression_ratio))
        self.non_depth_capture_bytes_per_s = max(0.0, float(non_depth_capture_bytes_per_s))
        self.max_tracking_age_s = float(max_tracking_age_s)
        if not math.isfinite(self.max_tracking_age_s) or self.max_tracking_age_s <= 0:
            raise ValueError("max_tracking_age_s must be finite and positive")
        self.capture_queue_frames = int(capture_queue_frames)
        if self.capture_queue_frames <= 0:
            raise ValueError("capture_queue_frames must be positive")
        self.camera_delivery_frames = int(camera_delivery_frames)
        if self.camera_delivery_frames <= 0:
            raise ValueError("camera_delivery_frames must be positive")
        self.camera_reopen_interval_s = float(camera_reopen_interval_s)
        if (
            not math.isfinite(self.camera_reopen_interval_s)
            or self.camera_reopen_interval_s <= 0
        ):
            raise ValueError("camera_reopen_interval_s must be finite and positive")
        self.audio_device = audio_device
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_channels = int(audio_channels)
        self.audio_pick_channel = (
            None if audio_pick_channel is None else int(audio_pick_channel)
        )
        self.audio_expected_usb_vid = audio_expected_usb_vid
        self.audio_expected_usb_pid = audio_expected_usb_pid
        self.audio_expected_usb_serial = audio_expected_usb_serial
        self.audio_expected_usb_port_path = audio_expected_usb_port_path
        self.require_audio = bool(require_audio)
        self.hall_calibration = hall_calibration
        self.hall_calibration_sha256 = hall_calibration_sha256
        self.hall_port = hall_port
        self.hall_expected_usb_vid = hall_expected_usb_vid
        self.hall_expected_usb_pid = hall_expected_usb_pid
        self.hall_expected_usb_serial = hall_expected_usb_serial
        self.sync_calibration = sync_calibration
        self.sync_calibration_sha256 = sync_calibration_sha256
        self.require_force = (
            not self.fake if require_force is None else bool(require_force)
        )
        if self.commissioning_hall_monitor_only and self.require_force:
            raise ValueError(
                "unsynchronized Hall monitor cannot satisfy required force capture"
            )
        self._force_requested = bool(
            self.require_force or self.commissioning_hall_monitor_only
        )
        self.require_robot = (
            not self.fake if require_robot is None else bool(require_robot)
        )
        self._robot_coordinator = robot_capture_coordinator
        # Optional trusted controller companion: the collector becomes the single
        # owner of the left-arm controller lifecycle, so the arm can be brought up
        # and energy-isolated from the console (no CLI).  Absent in preview/fake or
        # when no launch/disable command is configured -> the arm UI is inert.
        self._controller_launcher = controller_launcher
        self._controller_disable = controller_disable
        # P0-1: optional synchronous preflight (clear_faults + socket dir) run before the
        # controller binary is launched directly, so its exe attestation is sound.
        self._controller_preflight = controller_preflight
        self._controller_attest = controller_attest
        self._controller_readiness_timeout_s = float(controller_readiness_timeout_s)
        self._controller_companion = None
        self._controller_lock = threading.Lock()
        # A bring-up runs OUTSIDE the lock (so a concurrent E-stop can interrupt it);
        # this guards against a second concurrent bring-up.
        self._controller_bringup_active = False
        # Independent motor-disable readback hook wired into every companion. Absent
        # (None) -> energy_isolated is NEVER claimed (the honest default); a real
        # STATE-frame torque-off readback is injected here at deployment to earn it.
        # It is a constructor parameter (not a hardcoded None) so a production assembly
        # can wire a genuine per-motor torque-off decode without editing this class; the
        # decode source itself is a go-live wiring item (the STATE frame is not yet
        # decoded into per-motor torque flags).
        self._controller_verify_motor_disable = controller_verify_motor_disable
        # Independent controller watchdog cadence: supervision runs on its OWN thread so
        # a hung/slow main loop can never stall the telemetry-fault -> disable path.
        self._controller_watchdog_interval_s = 0.1
        self._controller_watchdog = None
        self._robot_mask_provider = robot_mask_provider
        self._handeye_activation = handeye_activation
        self._frame_evidence_bundle_factory = (
            FrameEvidenceBundle.build
            if frame_evidence_bundle_factory is None
            else frame_evidence_bundle_factory
        )
        self._back_mask_provider = back_mask_provider
        self._patient_pose_provider = patient_pose_provider
        self._occlusion_runtime = occlusion_runtime
        self._occlusion_episode_id = "preview-" + os.urandom(16).hex()
        self.robot_camera_arm_timeout_s = float(robot_camera_arm_timeout_s)
        if (
            not math.isfinite(self.robot_camera_arm_timeout_s)
            or self.robot_camera_arm_timeout_s <= 0.0
            or self.robot_camera_arm_timeout_s > 10.0
        ):
            raise ValueError(
                "robot camera-arm timeout must be within (0, 10] seconds"
            )
        self._robot_status = (
            "DISABLED" if not self.require_robot else "WAITING"
        )
        self._robot_error = None
        self._robot_last_summary = None
        if self.audio_sample_rate <= 0 or self.audio_channels <= 0:
            raise ValueError("audio sample rate and channel count must be positive")
        if (
            self.audio_pick_channel is not None
            and not 0 <= self.audio_pick_channel < self.audio_channels
        ):
            raise ValueError("audio pick channel must be within the opened channels")
        self._camera_factory = camera_factory
        self._processor_factory = processor_factory
        self._audio_factory = audio_factory
        self._audio_resolver = audio_resolver
        self._force_factory = force_factory
        self._hall_resolver = hall_resolver
        self._sync_mode_builder = sync_mode_builder
        self._sync_loader = sync_loader

        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        # End drains CameraStream and recursively submits each frame while owning
        # this boundary. RLock keeps that atomic without a second lock order.
        self._ingress_submit_lock = threading.RLock()
        # Capture ingress is armed only inside an explicit Start/End interval.
        # This prevents a pre-Start pending model frame from leaking into a new
        # session and creates a drainable, exact End boundary.
        self._capture_ingress_paused = True
        self._stop = threading.Event()
        self._thread = None
        self._loop_error = None
        # Health epoch: the loop bumps this each time the strict runtime liveness
        # transitions from healthy to degraded.  Start captures it before arming
        # and re-compares it at the final arm, so a worker that dies at any point
        # during session creation / capture.start() / sensor init forces a full
        # rollback instead of arming an empty recording.
        self._health_generation = 0
        self._last_runtime_strict_ok = True
        # Serialises the health-epoch bump (a worker death, observed on the loop
        # thread) against the final Start arm (on the request thread), so the
        # epoch check and the capture-ingress open are one atomic step.
        self._health_epoch_lock = threading.Lock()
        self._cam = None
        self._audio = None
        self._audio_hardware = None
        self._audio_status = "DISABLED" if self.audio_device is None else "STARTING"
        self._audio_error = None
        self._audio_recording_active = False
        self._audio_last_summary = None
        self._force = None
        self._force_hardware = None
        self._force_status = "DISABLED" if not self._force_requested else "STARTING"
        self._force_error = None
        self._force_hard_limit_n = None
        self._force_recording_active = False
        self._force_last_summary = None
        self._force_sync_metadata = None
        self._model_worker = None
        self._back_gate = None
        self._preview_worker = LatestFrameWorker(
            self._render_preview_bundle, name="wam-diagnostic-preview",
        ).start()
        self._preview_result_index = -1
        self._preview_publish_times = deque(maxlen=120)
        self._next_camera_open_t = 0.0
        self._ingress_worker = BoundedFifoWorker(
            self._ingest_bundle,
            capacity=self.capture_queue_frames,
            name="wam-diagnostic-ingress",
        ).start()
        self._frame_ring = OrderedDict()
        self._last_camera_index = -1
        self._last_result_index = -1
        self._record_last_camera_index = None
        self._latest_tracking = {
            "frame_index": 0,
            "back_present": False,
            "initialized": False,
            "detector_confidence": 0.0,
            "reason": "MODEL_STARTING",
            "points": [],
        }
        self._current_tracking = {
            **self._latest_tracking,
            "constellation_valid": False,
        }
        self._fatal_error = None
        self._camera_status = "STARTING"
        self._model_status = "SYNTHETIC" if self.fake else "STARTING"
        self._preview_frame_index = 0
        self._preview_frame_t = None
        self._preview_frame_confirmable = False
        self._preview_generation = 0
        self._preview_token_key = os.urandom(32)
        self._current_depth_plausible = bool(self.fake)
        self._record_started = None
        self._record_stopped = None
        self._recording_boundary = None
        self._end_boundary_closing = False
        self._record_ingress_drop_baseline = 0
        self._record_ingress_overflow_baseline = 0
        self._record_camera_overflow_baseline = 0
        self._camera_delivery_interval_id = None
        self._camera_delivery_enabled = False
        self._pending_delivery_frame = None
        self._latest_recorded_robot_evidence = None
        self._pending_model_frame = None
        self._pending_model_preview = None
        self._completion_proof_cache = None
        self.jpeg = encode_jpeg(np.zeros((480, 640, 3), np.uint8))

        missing = self._configuration_blockers()
        self._diagnostic_blockers = tuple(missing or (["camera/model starting"] if not self.fake else []))
        self._provenance = {
            "camera": {"backend": self.camera_backend, "camera_id": self.camera_id or "synthetic"},
            "model": {
                "release_id": self.model_release_id or "unavailable",
                "checkpoint_sha256": self.checkpoint_sha256 or "unavailable",
                "base_model_sha256": self.base_model_sha256 or "unavailable",
                "posterior_gate_release_id": self.posterior_gate_release_id or "unavailable",
                "posterior_gate_sha256": (
                    self.posterior_gate_checkpoint_sha256 or "unavailable"
                ),
            },
            "runtime": {
                "release_id": self.runtime_release_id or "unavailable",
                "lock_sha256": self.runtime_lock_sha256 or "unavailable",
                "code_release_id": self.code_release_id or "unavailable",
                "ffmpeg_path": self.ffmpeg_path or "unavailable",
                "ffmpeg_sha256": self.ffmpeg_sha256 or "unavailable",
            },
            "audio": {
                "device": self.audio_device or "disabled",
                "input_sample_rate": self.audio_sample_rate,
                "model_sample_rate": 16_000,
                "channels": self.audio_channels,
                "pick_channel": self.audio_pick_channel,
                "hardware": None,
                "single_owner": True,
            },
            "force": {
                "single_owner": True,
                "mode": (
                    "UNSYNCED_COMMISSIONING_MONITOR"
                    if self.commissioning_hall_monitor_only
                    else "SYNCHRONIZED_CAPTURE"
                ),
                "hardware": None,
                "calibration_path": self.hall_calibration,
                "calibration_sha256": self.hall_calibration_sha256,
                "sync_calibration_path": self.sync_calibration,
                "sync_calibration_sha256": self.sync_calibration_sha256,
                "sync": None,
            },
        }
        # The console carries a preview, diagnostic, or -- only with --production-teleop
        # and a live robot -- a production contract. The writer_types binding is
        # non-forgeable (identity string -> exact class object), so a session can only
        # record with the exact writer its contract names.
        self.platform = CollectionPlatform(
            session_root=self.out_dir,
            point_names=self.point_names,
            capture_factory=self._capture_factory,
            contract=(
                PRODUCTION_TELEOP_CONTRACT
                if self.production_teleop
                else PREVIEW_ONLY_CONTRACT
                if self.commissioning_preview_only
                else DIAGNOSTIC_PERCEPTION_CONTRACT
            ),
            writer_types={
                "DiagnosticPerceptionCapture": DiagnosticPerceptionCapture,
                "ProductionTeleopCapture": ProductionTeleopCapture,
            },
            diagnostic_blockers=self._diagnostic_blockers,
            production_blockers=tuple(production_blockers),
            # The production platform's contract-driven End validation + the writer's own
            # machinery are the gates for production; the diagnostic-seal completion
            # validator applies only to diagnostic sessions.
            completion_validator=(
                None if self.production_teleop else self._validate_completion_result
            ),
        )
        if start_thread:
            self.start()

    def _configuration_blockers(self):
        if self.fake:
            return []
        required = {
            "stable Orbbec camera serial": self.camera_id,
            "pinned Orbbec color width": self.camera_width,
            "pinned Orbbec color height": self.camera_height,
            "pinned Orbbec color FPS": self.camera_fps,
            "pinned Orbbec color format": self.camera_color_format,
            "pinned Orbbec depth width": self.depth_width,
            "pinned Orbbec depth height": self.depth_height,
            "pinned Orbbec depth FPS": self.depth_fps,
            "pinned Orbbec depth format": self.depth_format,
            "v9 checkpoint path": self.checkpoint,
            "v9 checkpoint SHA-256": self.checkpoint_sha256,
            "DINOv2 base path": self.base_model,
            "DINOv2 base SHA-256": self.base_model_sha256,
            "perception model release id": self.model_release_id,
            "posterior-person gate checkpoint path": self.posterior_gate_checkpoint,
            "posterior-person gate checkpoint SHA-256": self.posterior_gate_checkpoint_sha256,
            "posterior-person gate release id": self.posterior_gate_release_id,
            "runtime release id": self.runtime_release_id,
            "runtime lock path": self.runtime_lock,
            "runtime lock SHA-256": self.runtime_lock_sha256,
            "code release id": self.code_release_id,
            "pinned ffmpeg concat-copy path": self.ffmpeg_path,
            "pinned ffmpeg concat-copy SHA-256": self.ffmpeg_sha256,
        }
        if self.require_audio:
            required["commissioned stethoscope audio device"] = self.audio_device
            required["stethoscope USB VID"] = self.audio_expected_usb_vid
            required["stethoscope USB PID"] = self.audio_expected_usb_pid
        if self._force_requested:
            required.update({
                "calibrated Hall-force calibration path": self.hall_calibration,
                "Hall-force calibration SHA-256": self.hall_calibration_sha256,
                "commissioned Hall-force serial port": self.hall_port,
                "Hall-force USB VID": self.hall_expected_usb_vid,
                "Hall-force USB PID": self.hall_expected_usb_pid,
                "Hall-force USB serial": self.hall_expected_usb_serial,
            })
        if self.require_force:
            required.update({
                "physical audio-force sync calibration path": self.sync_calibration,
                "physical audio-force sync calibration SHA-256": self.sync_calibration_sha256,
            })
        blockers = [
            f"missing {label}" for label, value in required.items()
            if value is None or value == ""
        ]
        if self.require_force and not self.require_audio:
            blockers.append(
                "Hall-force synchronization requires commissioned stethoscope audio"
            )
        if self.require_force and self.audio_pick_channel is None:
            blockers.append(
                "Hall-force synchronization requires a pinned audio channel"
            )
        if (
            self.require_audio
            and not self.audio_expected_usb_serial
            and not self.audio_expected_usb_port_path
        ):
            blockers.append("missing stethoscope USB serial or fixed topology")
        if self.require_robot:
            if self._robot_coordinator is None:
                blockers.append("missing left OpenArm capture coordinator")
            if self._robot_mask_provider is None:
                blockers.append(
                    "missing commissioned OpenArm robot-mask provider"
                )
            if not callable(self._frame_evidence_bundle_factory):
                blockers.append("missing same-frame robot evidence builder")
            if self._back_mask_provider is None:
                blockers.append(
                    "missing independent posterior back-mask provider"
                )
            if self._patient_pose_provider is None:
                blockers.append("missing metric patient-pose provider")
            if self._occlusion_runtime is None:
                blockers.append("missing occlusion safety runtime")
        return blockers

    def _capture_factory(self, session_dir):
        capture_cls = (
            ProductionTeleopCapture if self.production_teleop
            else DiagnosticPerceptionCapture
        )
        return capture_cls(
            session_dir,
            fps=self._negotiated_camera_fps(),
            provenance=self._provenance,
            recording_boundary=self._recording_boundary,
            ffmpeg_path=self.ffmpeg_path,
            ffmpeg_sha256=self.ffmpeg_sha256,
            min_free_bytes=self.diagnostic_min_free_bytes,
            depth_chunk_frames=5,
            checkpoint_every_frames=5,
            require_depth=True,
            require_robot=self.require_robot,
            require_force=self.require_force,
        )

    def _negotiated_camera_fps(self):
        mode = self._provenance.get("camera", {}).get("negotiated_mode")
        if isinstance(mode, dict) and "fps" in mode:
            try:
                fps = float(mode["fps"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("negotiated camera FPS is invalid") from exc
            if not math.isfinite(fps) or fps <= 0:
                raise RuntimeError("negotiated camera FPS is invalid")
            return fps
        if self.fake:
            return self.diagnostic_fps
        raise RuntimeError("negotiated camera FPS is unavailable")

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        if self.audio_device is not None and self._audio is None:
            try:
                audio_open_device = self.audio_device
                if (
                    self.audio_expected_usb_vid is not None
                    or self.audio_expected_usb_pid is not None
                    or self.audio_expected_usb_serial is not None
                    or self.audio_expected_usb_port_path is not None
                ):
                    if self._audio_resolver is None:
                        from record_session import resolve_audio_hardware_descriptor
                        resolver = resolve_audio_hardware_descriptor
                        observed = resolver(
                            self.audio_device,
                            expected_usb_vid=self.audio_expected_usb_vid,
                            expected_usb_pid=self.audio_expected_usb_pid,
                            expected_usb_serial=self.audio_expected_usb_serial,
                        )
                    else:
                        resolver = self._audio_resolver
                        observed = resolver(self.audio_device)
                    self._audio_hardware = validate_audio_hardware_descriptor(
                        observed,
                        expected_vid=self.audio_expected_usb_vid,
                        expected_pid=self.audio_expected_usb_pid,
                        expected_serial=self.audio_expected_usb_serial,
                        expected_port_path=self.audio_expected_usb_port_path,
                        channels=self.audio_channels,
                        pick_channel=self.audio_pick_channel,
                    )
                    self._provenance["audio"]["hardware"] = copy.deepcopy(
                        self._audio_hardware,
                    )
                    audio_open_device = self._audio_hardware["portaudio_index"]
                if self._audio_factory is None:
                    from ausc.audio.audio_stream import AudioStream
                    factory = AudioStream
                else:
                    factory = self._audio_factory
                self._audio = factory(
                    device=audio_open_device,
                    sr_in=self.audio_sample_rate,
                    sr_model=16_000,
                    channels=self.audio_channels,
                    pick_channel=self.audio_pick_channel,
                )
                self._audio.start()
                self._audio_status = "RUNNING"
                self._audio_error = None
            except Exception as exc:  # noqa: BLE001 - surface hardware failure in UI
                self._audio = None
                self._audio_status = "ERROR"
                self._audio_error = f"{type(exc).__name__}: {exc}"
                if self.require_audio:
                    self._set_diagnostic_blockers([
                        f"stethoscope audio failed to start: {self._audio_error}",
                    ])
        if self._force_requested and self._force is None:
            try:
                calibration_path, calibration_sha, _, _ = _verified_immutable_input(
                    self.hall_calibration, label="Hall-force calibration",
                )
                if calibration_sha != self.hall_calibration_sha256:
                    raise RuntimeError("Hall-force calibration SHA-256 mismatch")
                from mechanical_sensor.force_feedback.force_sensor import ForceCalibration
                force_calibration = ForceCalibration.from_json(calibration_path)
                if force_calibration.calibrated is not True:
                    raise RuntimeError(
                        "Hall-force calibration is not physically commissioned"
                    )
                self._force_hard_limit_n = float(
                    force_calibration.hard_limit_force_n
                )
                from ausc.record_session import (
                    build_sync_capture_mode,
                    load_sync_calibration,
                    resolve_hall_hardware_descriptor,
                )
                hall_resolver = self._hall_resolver or resolve_hall_hardware_descriptor
                observed = hall_resolver(self.hall_port)
                if not isinstance(observed, Mapping):
                    raise RuntimeError("Hall-force hardware descriptor is invalid")
                expected = {
                    "usb_vid": self.hall_expected_usb_vid,
                    "usb_pid": self.hall_expected_usb_pid,
                    "usb_serial": self.hall_expected_usb_serial,
                }
                for field, value in expected.items():
                    if observed.get(field) != value:
                        raise RuntimeError(f"Hall-force {field} identity mismatch")
                if observed.get("requested_port") != self.hall_port:
                    raise RuntimeError("Hall-force requested serial identity mismatch")
                sync_path = None
                sync_sha = None
                if self.require_force:
                    sync_path, sync_sha, _, _ = _verified_immutable_input(
                        self.sync_calibration,
                        label="audio-force sync calibration",
                    )
                    if sync_sha != self.sync_calibration_sha256:
                        raise RuntimeError(
                            "audio-force sync calibration SHA-256 mismatch"
                        )
                    if not isinstance(self._audio_hardware, Mapping):
                        raise RuntimeError(
                            "audio-force sync requires the exact opened audio hardware identity"
                        )
                    mode_builder = self._sync_mode_builder or build_sync_capture_mode
                    capture_mode = mode_builder(
                        audio_device=self.audio_device,
                        audio_channel=self.audio_pick_channel,
                        hall_port=self.hall_port,
                        force_calibration_sha256=calibration_sha,
                        hall_resolver=lambda _port: copy.deepcopy(dict(observed)),
                        audio_resolver=lambda _device: copy.deepcopy(
                            dict(self._audio_hardware)
                        ),
                    )
                    sync_loader = self._sync_loader or load_sync_calibration
                    sync_metadata = sync_loader(
                        sync_path, expected_capture_mode=capture_mode,
                    )
                    if not isinstance(sync_metadata, Mapping):
                        raise RuntimeError(
                            "audio-force sync calibration metadata is invalid"
                        )
                    self._force_sync_metadata = json.loads(
                        json.dumps(sync_metadata)
                    )
                else:
                    self._force_sync_metadata = {
                        "schema": "wam_commissioning_unsynced_force_v1",
                        "status": "UNSYNCED_COMMISSIONING_ONLY",
                        "synchronized": False,
                        "training_eligible": False,
                    }
                if self._force_factory is None:
                    from mechanical_sensor.force_feedback.force_capture import ForceCaptureStream
                    force_factory = ForceCaptureStream
                else:
                    force_factory = self._force_factory
                self._force = force_factory(
                    self.hall_port,
                    calibration_path,
                    device_identity=dict(observed),
                    on_safety_event=self._on_force_safety_event,
                    on_capture_fault=self._on_force_capture_fault,
                )
                calibration_identity = getattr(
                    self._force, "calibration_identity", None
                )
                if (
                    isinstance(calibration_identity, Mapping)
                    and calibration_identity.get("sha256") != calibration_sha
                ):
                    raise RuntimeError(
                        "Hall-force owner opened different calibration bytes"
                    )
                self._force.prepare()
                self._force_hardware = copy.deepcopy(dict(observed))
                self._provenance["force"] = {
                    "single_owner": True,
                    "mode": (
                        "SYNCHRONIZED_CAPTURE"
                        if self.require_force
                        else "UNSYNCED_COMMISSIONING_MONITOR"
                    ),
                    "hardware": copy.deepcopy(dict(observed)),
                    "calibration_path": str(calibration_path),
                    "calibration_sha256": calibration_sha,
                    "hard_limit_force_n": self._force_hard_limit_n,
                    "sync_calibration_path": (
                        None if sync_path is None else str(sync_path)
                    ),
                    "sync_calibration_sha256": sync_sha,
                    "sync": copy.deepcopy(self._force_sync_metadata),
                }
                self._force_status = "RUNNING"
                self._force_error = None
            except Exception as exc:  # noqa: BLE001 - requested sensor fails closed
                self._force = None
                self._force_status = "ERROR"
                self._force_error = f"{type(exc).__name__}: {exc}"
                self._set_diagnostic_blockers([
                    f"Hall-force failed to start: {self._force_error}",
                ])
        self._loop_error = None
        self._thread = threading.Thread(
            target=self._loop_guarded,
            name="wam-collection-preview",
            daemon=True,
        )
        self._thread.start()
        # Independent controller watchdog: supervises the arm on its OWN cadence so a
        # hung/slow main loop can never stall the telemetry-fault -> disable path.
        self._controller_watchdog = threading.Thread(
            target=self._controller_watchdog_run,
            name="wam-controller-watchdog",
            daemon=True,
        )
        self._controller_watchdog.start()
        return self

    def _controller_watchdog_run(self):
        # Sole controller supervisor, independent of the main loop. A telemetry fault or
        # a controller exit after ready energy-isolates the arm (poll() disable-firsts).
        # receiver_health() is non-blocking (cached snapshot + staleness), so this can
        # neither hang nor be starved by main-loop work.
        while not self._stop.wait(self._controller_watchdog_interval_s):
            # This is the SOLE controller supervisor. Its per-tick body is wrapped so that
            # NOTHING -- not even a BaseException escaping poll()/shutdown() -- can kill the
            # thread and silently leave the arm unsupervised. Any escape fail-safe-disables
            # and the loop continues; clean shutdown is still driven only by self._stop.
            try:
                self._watchdog_tick()
            except BaseException as exc:  # noqa: BLE001 - the supervisor must never die
                comp = self._controller_companion
                try:
                    if comp is not None and not comp.disabled:
                        comp.report_external_fault(
                            f"watchdog tick error: {type(exc).__name__}: {exc}"
                        )
                except Exception:
                    pass

    def _watchdog_tick(self):
        comp = self._controller_companion
        if comp is None or comp.disabled:
            return
        if comp.state == "ready":
            try:
                comp.poll()
            except CompanionFault:
                pass  # poll() already disabled the arm; status reflects it
            except Exception as exc:  # noqa: BLE001 - any watchdog error -> fail-safe disable
                if not comp.disabled:
                    comp.report_external_fault(
                        f"controller watchdog error: {type(exc).__name__}: {exc}"
                    )
        elif comp.state == "faulted" and comp.controller_pid is not None:
            # Faulted with a launched controller but NO confirmed disable -> a prior
            # disable command failed. Re-issue it every tick until it succeeds, so a
            # single transient disable failure can never leave the arm energized.
            # shutdown() is retryable (it latches _disabled only on command success)
            # and a no-op once the command finally confirms.
            comp.shutdown("watchdog_disable_retry")

    def _loop_guarded(self):
        try:
            self._loop()
        except BaseException as exc:  # a dead orchestration loop is a P0 fault
            reason = f"collector loop failed: {type(exc).__name__}: {exc}"
            # SAFETY FIRST (disable-first): a dead collector loop means the arm's
            # supervisor is gone. Energy-isolate the arm BEFORE any data teardown, so a
            # failure inside _abort_recording() (or anywhere below) can never leave the
            # arm live under teleop. A real software safety fault DOES stop the arm
            # (distinct from an advisory force reading).
            comp = self._controller_companion
            if comp is not None and not comp.disabled:
                try:
                    comp.report_external_fault(reason)
                except Exception:  # noqa: BLE001 - disable must never be masked
                    pass
            self._loop_error = reason
            self._fatal_error = reason
            self._camera_status = "ERROR"
            self._model_status = "ERROR"
            # Data teardown is guarded so it can neither undo the disable above nor
            # re-raise out of the loop thread.
            try:
                failed = {
                    "frame_index": int(self._last_camera_index),
                    "back_present": False,
                    "initialized": False,
                    "constellation_valid": False,
                    "reason": "COLLECTOR_LOOP_ERROR",
                    "points": [],
                }
                with self._lock:
                    self._current_tracking = failed
                self.platform.update_tracking(failed)
                self._set_diagnostic_blockers([reason])
                self._abort_recording(reason)
            except Exception:  # noqa: BLE001 - teardown must not undo the disable
                pass

    def shutdown(self):
        self._clear_back_confirmation("SERVICE_SHUTDOWN")
        self._stop.set()
        if self.platform.state is CollectionState.RECORDING:
            self._abort_recording("service shutdown before End")
        # Never leave the arm energized across a collector shutdown. The watchdog has just
        # been stopped (_stop.set above), so it can no longer retry a failed disable --
        # retry here, bounded, rather than swallowing a single transient failure.
        if self._controller_companion is not None and not self._controller_companion.disabled:
            for _ in range(5):
                try:
                    if self.disable_controller("service_shutdown").get("ok"):
                        break
                except Exception:  # noqa: BLE001 - keep retrying the shutdown disable
                    pass
        if self._model_worker is not None:
            self._model_worker.stop()
        if self._preview_worker is not None:
            self._preview_worker.stop()
        if self._ingress_worker is not None:
            self._ingress_worker.stop()
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass
        if self._audio is not None:
            try:
                self._audio.stop()
            except Exception:
                pass
            finally:
                self._audio_recording_active = False
        if self._force is not None:
            try:
                self._force.stop()
            except Exception:
                pass
            finally:
                self._force_recording_active = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        # Always drain the one non-daemon finalizer thread, even if the session
        # was Aborted out of FINALIZING while the thread was still mid-flight;
        # otherwise the interpreter cannot exit cleanly.  A no-op when the thread
        # is already gone.  systemd owns the outer stop timeout for a genuinely
        # stuck finalize.
        self.platform.wait_for_terminal(timeout=None)

    def control(
        self, action, *, preview_frame_index=None, preview_frame_token=None,
    ):
        # End owns this lock through FIFO closure, exact drain and the platform
        # transition to FINALIZING. No Confirm/Redo/navigation can slip through
        # the gap between the camera boundary and finalization.
        with self._control_lock:
            return self._control_locked(
                action,
                preview_frame_index=preview_frame_index,
                preview_frame_token=preview_frame_token,
            )

    def _back_confirmation_snapshot(self):
        gate = self._back_gate
        if gate is None:
            if self.fake:
                return {
                    "required": False,
                    "mode": "synthetic",
                    "status": "ACTIVE",
                    "confirmable": False,
                    "confirmed": True,
                    "active": True,
                    "reason": "SYNTHETIC_BACK",
                    "person_frame_index": self._last_result_index,
                    "confirmation_frame_index": None,
                    "missed_frames": 0,
                }
            return {
                "required": True,
                "mode": "operator",
                "status": "UNAVAILABLE",
                "confirmable": False,
                "confirmed": False,
                "active": False,
                "reason": "PERSON_GATE_STARTING",
                "person_frame_index": None,
                "confirmation_frame_index": None,
                "missed_frames": 0,
            }
        try:
            snapshot = gate.confirmation_snapshot()
        except Exception as exc:  # noqa: BLE001 - control state must fail closed
            return {
                "required": True,
                "mode": "operator",
                "status": "ERROR",
                "confirmable": False,
                "confirmed": False,
                "active": False,
                "reason": f"BACK_GATE_ERROR: {type(exc).__name__}: {exc}",
                "person_frame_index": None,
                "confirmation_frame_index": None,
                "missed_frames": 0,
            }
        result = dict(snapshot) if isinstance(snapshot, dict) else {}
        result.setdefault("required", True)
        result.setdefault("mode", "operator")
        result.setdefault("confirmed", False)
        result.setdefault("active", False)
        result.setdefault("confirmable", False)
        return result

    def _clear_back_confirmation(self, reason):
        gate = self._back_gate
        if gate is None:
            return None
        clear = getattr(gate, "clear_confirmation", None)
        if not callable(clear):
            clear = getattr(gate, "clear", None)
        if not callable(clear):
            return None
        try:
            return clear(str(reason))
        except Exception:
            # The visible and Start gates still fail closed through the snapshot.
            return None

    @staticmethod
    def _robot_permit_payload(permit):
        if is_dataclass(permit):
            payload = asdict(permit)
        elif isinstance(permit, Mapping):
            payload = dict(permit)
        elif hasattr(permit, "__dict__"):
            payload = dict(vars(permit))
        else:
            raise RuntimeError("robot camera-arm permit is not serializable")
        if payload.get("pair") != "left:can1->can3":
            raise RuntimeError(
                "robot camera-arm permit is not left can1 leader -> can3 follower"
            )
        try:
            authorized_ns = int(payload["authorized_monotonic_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "robot camera-arm permit has no monotonic authorization"
            ) from exc
        if authorized_ns <= 0:
            raise RuntimeError(
                "robot camera-arm permit has invalid monotonic authorization"
            )
        payload["authorized_monotonic_ns"] = authorized_ns
        return payload

    def _controller_configured(self):
        """True when the console can own the arm controller lifecycle."""
        return bool(
            self.require_robot
            and not self.commissioning_preview_only
            and self._robot_coordinator is not None
            and callable(self._controller_launcher)
            and callable(self._controller_disable)
        )

    def _controller_ready(self):
        comp = self._controller_companion
        return bool(comp is not None and comp.state == "ready" and not comp.disabled)

    def bring_up_controller(self):
        """Console-triggered left-arm bring-up via the trusted companion.

        Single owner; a fresh companion per bring-up; fault-closed to a
        direct-disable.  Real motion begins here, so it is gated to a configured,
        non-preview collector and returns only after the receiver reports per-motor
        READY (or after the arm has been energy-isolated on failure).
        """
        if not self._controller_configured():
            return {
                "ok": False, "code": "controller_not_configured",
                "message": "arm controller launch/disable is not configured on this collector",
            }
        # A fatal collector loop (or a service shutdown in progress) means the arm's
        # supervisor is gone / going. Refuse a fresh bring-up onto a dead/stopping owner.
        if (
            self._fatal_error is not None
            or self._loop_error is not None
            or self._stop.is_set()
        ):
            return {
                "ok": False, "code": "controller_bring_up_refused_fatal",
                "message": (
                    "collector is in a fatal/stopping state; arm bring-up refused: "
                    f"{self._fatal_error or self._loop_error or 'service shutdown'}"
                ),
            }
        with self._controller_lock:
            comp = self._controller_companion
            if comp is not None and comp.state == "ready" and not comp.disabled:
                return {
                    "ok": True, "code": "controller_already_up",
                    "message": "left arm controller already brought up",
                    "controller_pid": comp.controller_pid,
                }
            if self._controller_bringup_active:
                return {
                    "ok": False, "code": "controller_bring_up_in_progress",
                    "message": "an arm bring-up is already in progress",
                }
            # SAFETY: never replace an outgoing companion whose controller was launched
            # but is NOT confirmed disabled (e.g. a prior disable command failed and the
            # watchdog is still retrying it). Overwriting self._controller_companion here
            # would ORPHAN that companion -- the watchdog and the E-stop both follow
            # self._controller_companion, so #1's disable-retry loop would be abandoned and
            # a SECOND controller would be launched onto a non-isolated arm. Refuse until
            # the arm is confirmed energy-isolated (the watchdog is retrying) or E-stopped.
            # A companion with no launched controller (controller_pid is None, e.g. a
            # preflight failure) has nothing live to orphan, so it may be replaced.
            if (
                comp is not None
                and not comp.disabled
                and comp.controller_pid is not None
            ):
                return {
                    "ok": False, "code": "controller_prior_not_isolated",
                    "message": (
                        "the previous arm controller is not confirmed disabled; wait for "
                        "it to be energy-isolated (the watchdog is retrying) or press "
                        "E-stop before bringing up again"
                    ),
                    "energy_isolated": False,
                }
            companion = TrustedControllerCompanion(
                coordinator=self._robot_coordinator,
                launch_controller=self._controller_launcher,
                direct_disable=self._controller_disable,
                preflight=self._controller_preflight,
                attest_controller=self._controller_attest,
                verify_motor_disable=self._controller_verify_motor_disable,
                readiness_timeout_s=self._controller_readiness_timeout_s,
            )
            self._controller_companion = companion
            self._controller_bringup_active = True
        # Run bring_up OUTSIDE the lock so a concurrent E-stop (disable_controller) can
        # acquire the lock and interrupt it -- the companion aborts its readiness wait
        # the moment it is disabled, and never launches onto a disabled owner.
        try:
            companion.bring_up()
        except CompanionFault as exc:
            outcome = companion.disable_outcome
            isolated = bool(outcome is not None and outcome.energy_isolated)
            detail = (
                "energy-isolated" if isolated
                else "disable commanded, MOTOR-DISABLE UNVERIFIED"
            )
            result = {
                "ok": False, "code": "controller_bring_up_faulted",
                "message": f"arm bring-up failed and was {detail}: {exc}",
                "energy_isolated": isolated,
                "motor_disable_verified": bool(
                    outcome is not None and outcome.motor_verified
                ),
            }
        except CompanionError as exc:
            result = {"ok": False, "code": "controller_bring_up_error", "message": str(exc)}
        else:
            # A collector-loop crash or service shutdown that raced in DURING this bring-up
            # leaves the arm's supervisor gone even though bring_up() reached READY. The
            # top-of-method check cannot cover this window (it runs before launch), so
            # re-check here and energy-isolate rather than returning a live-arm success onto
            # a dead/stopping collector.
            if (
                self._fatal_error is not None
                or self._loop_error is not None
                or self._stop.is_set()
            ):
                try:
                    companion.report_external_fault(
                        "collector became fatal/stopping during bring-up"
                    )
                except Exception:  # noqa: BLE001 - disable must never be masked
                    pass
                outcome = companion.disable_outcome
                result = {
                    "ok": False, "code": "controller_bring_up_refused_fatal",
                    "message": (
                        "collector went fatal or is shutting down during bring-up; arm "
                        "energy-isolated instead of brought up: "
                        f"{self._fatal_error or self._loop_error or 'service shutdown'}"
                    ),
                    "energy_isolated": bool(
                        outcome is not None and outcome.energy_isolated
                    ),
                }
            else:
                result = {
                    "ok": True, "code": "controller_ready",
                    "message": "left arm controller ready (per-motor telemetry verified)",
                    "controller_pid": companion.controller_pid,
                }
        finally:
            with self._controller_lock:
                self._controller_bringup_active = False
        return result

    def disable_controller(self, reason="operator_estop"):
        """Console E-stop: energy-isolate the arm (direct-disable first)."""
        with self._controller_lock:
            comp = self._controller_companion
            if comp is None:
                return {"ok": True, "code": "controller_not_running", "message": "no arm controller is running"}
            try:
                outcome = comp.shutdown(str(reason))
            except Exception as exc:  # noqa: BLE001 - disable path must report, never raise out
                return {"ok": False, "code": "controller_disable_error", "message": f"{type(exc).__name__}: {exc}"}
            # E-stop honesty: ok reflects whether the disable COMMAND actually succeeded.
            # If it failed (command_succeeded False), the operator must be told the arm
            # was NOT disabled -- never report success. Energy isolation is a further
            # claim that requires a verified motor-disable readback.
            if not outcome.command_succeeded:
                return {
                    "ok": False, "code": "controller_disable_command_failed",
                    "message": f"left arm DISABLE COMMAND FAILED — arm NOT confirmed disabled ({outcome.outcome})",
                    "energy_isolated": False,
                    "motor_disable_verified": bool(outcome.motor_verified),
                }
            message = (
                "left arm energy-isolated (motor disable verified)"
                if outcome.energy_isolated
                else f"left arm disable commanded — MOTOR-DISABLE UNVERIFIED ({outcome.outcome})"
            )
            return {
                "ok": True, "code": "controller_disabled", "message": message,
                "energy_isolated": bool(outcome.energy_isolated),
                "motor_disable_verified": bool(outcome.motor_verified),
            }

    def _controller_status_payload(self):
        comp = self._controller_companion
        if not self._controller_configured():
            return {"configured": False, "state": "unavailable"}
        if comp is None:
            return {"configured": True, "state": "idle"}
        payload = {
            "configured": True, "state": comp.state,
            "disabled": bool(comp.disabled), "controller_pid": comp.controller_pid,
        }
        # P0-2: surface the honest disable ladder. "energy_isolated" is only ever True
        # after an independent motor-disable readback; a mere command success shows as
        # command_succeeded + motor_disable_verified=False so the UI never lies.
        outcome = comp.disable_outcome
        if outcome is not None:
            payload.update({
                "disable_command_succeeded": bool(outcome.command_succeeded),
                "motor_disable_verified": bool(outcome.motor_verified),
                "energy_isolated": bool(outcome.energy_isolated),
                "disable_summary": str(outcome.outcome),
            })
        return payload

    def _start_robot_session(self, session_dir):
        if not self.require_robot:
            return None
        if self._controller_configured() and not self._controller_ready():
            raise RuntimeError(
                "left OpenArm controller is not brought up; use Bring up arm first"
            )
        coordinator = self._robot_coordinator
        if coordinator is None:
            raise RuntimeError("left OpenArm capture coordinator is unavailable")
        coordinator.start_session(session_dir)
        deadline = time.monotonic() + self.robot_camera_arm_timeout_s
        while True:
            try:
                permit = coordinator.authorize_camera_arm()
                payload = self._robot_permit_payload(permit)
                self._robot_status = "READY"
                self._robot_error = None
                return payload
            except RobotCaptureNotReady:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "left OpenArm post-Start telemetry did not authorize "
                        "camera arm before timeout"
                    )
                time.sleep(min(0.005, remaining))

    def _abort_robot_session(self, reason):
        coordinator = self._robot_coordinator
        if coordinator is None:
            return False
        abort = getattr(coordinator, "abort_session", None)
        if not callable(abort):
            if self.require_robot:
                self._robot_status = "ERROR"
                self._robot_error = (
                    "left OpenArm coordinator cannot abort an incomplete session"
                )
            return False
        try:
            aborted = bool(abort(str(reason)))
        except Exception as exc:  # cleanup continues; the sensor session stays invalid
            self._robot_status = "ERROR"
            self._robot_error = f"{type(exc).__name__}: {exc}"
            return False
        if aborted:
            self._robot_status = "WAITING"
            self._robot_last_summary = None
        return aborted

    @staticmethod
    def _actual_robot_activation_contract(coordinator):  # noqa: ANN205
        from ausc.perception.robot.robot_state_stream import RobotStateContract

        contract = getattr(coordinator, "contract", None)
        frozen = getattr(type(contract), "__dataclass_params__", None)
        if (
            type(contract) is not RobotStateContract
            or frozen is None
            or frozen.frozen is not True
        ):
            raise RuntimeError(
                "left OpenArm coordinator lacks the actual frozen RobotStateContract"
            )
        return contract

    def _finish_robot_session(self, *, expected_frame_count):
        if not self.require_robot:
            return None
        coordinator = self._robot_coordinator
        if coordinator is None:
            raise RuntimeError("left OpenArm capture coordinator is unavailable")
        contract = self._actual_robot_activation_contract(coordinator)
        summary = coordinator.finish_session()
        if not isinstance(summary, Mapping):
            raise RuntimeError("left OpenArm final summary is not a mapping")
        summary = dict(summary)
        if (
            not _robot_v4_summary_valid(summary, contract=contract)
            or summary.get("side") != "left"
            or summary.get("controller_mode") != "unilateral"
            or summary.get("leader_can_interface") != "can1"
            or summary.get("follower_can_interface") != "can3"
        ):
            raise RuntimeError(
                "left OpenArm final summary lacks clean v4 feedback liveness, "
                "trajectory authenticity, or SocketCAN TX evidence"
            )
        try:
            robot_frames = int(summary["frame_count"])
            rgbd_frames = int(expected_frame_count)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "robot/RGB-D frame counts are unavailable"
            ) from exc
        if (
            robot_frames <= 0
            or rgbd_frames <= 0
            or robot_frames != rgbd_frames
        ):
            raise RuntimeError(
                "robot/RGB-D frame count mismatch: "
                f"robot={robot_frames}, rgbd={rgbd_frames}"
            )
        capture = self.platform.capture
        setter = getattr(capture, "set_robot_summary", None)
        if not callable(setter):
            raise RuntimeError(
                "diagnostic capture cannot bind finalized robot evidence"
            )
        setter(summary, contract=contract)
        self._robot_last_summary = copy.deepcopy(summary)
        self._robot_status = "READY"
        self._robot_error = None
        return summary

    def _fail_start_after_platform(self, reason, *, code):
        """Abort every partially opened Start resource and clear authorization."""

        self._capture_ingress_paused = True
        self._ingress_worker.close_interval()
        self._abort_camera_delivery()
        if self._audio is not None:
            try:
                self._audio.abort_recording()
            except Exception:  # cleanup must continue after a partially opened audio writer
                pass
        self._audio_recording_active = False
        if self._force is not None:
            try:
                self._force.abort_recording()
            except Exception:
                pass
        self._force_recording_active = False
        self._abort_robot_session(reason)
        self.platform.abort(str(reason))
        self._clear_back_confirmation("START_FAILED")
        self._occlusion_episode_id = "preview-" + os.urandom(16).hex()
        self._recording_boundary = None
        self._pending_delivery_frame = None
        return {
            "ok": False,
            "code": str(code),
            "message": str(reason),
            "state": self.platform.state.value,
        }

    def _camera_delivery_health(self):
        camera = self._cam
        getter = getattr(camera, "delivery_health", None)
        if not callable(getter):
            return {
                "status": "UNAVAILABLE",
                "interval_active": False,
                "accepting": False,
                "queue_depth": 0,
                "capacity": 0,
                "interval_overflow_frames": 0,
                "overflow_frames": 0,
            }
        try:
            health = getter()
        except Exception as exc:  # noqa: BLE001 - source failure is surfaced, never hidden
            return {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "interval_active": False,
                "accepting": False,
                "queue_depth": 0,
                "capacity": self.camera_delivery_frames,
                "interval_overflow_frames": 0,
                "overflow_frames": 0,
            }
        return dict(health) if isinstance(health, dict) else {
            "status": "ERROR",
            "error": "camera delivery health is not a mapping",
            "queue_depth": 0,
            "capacity": self.camera_delivery_frames,
            "interval_overflow_frames": 0,
            "overflow_frames": 0,
        }

    def _abort_camera_delivery(self):
        camera = self._cam
        aborter = getattr(camera, "abort_delivery_interval", None)
        if callable(aborter):
            try:
                aborter()
            except Exception:
                # The lifecycle is already fail-closed; retain the primary reason.
                pass
        self._camera_delivery_enabled = False
        self._camera_delivery_interval_id = None
        self._pending_delivery_frame = None
        self._latest_recorded_robot_evidence = None

    def _camera_delivery_overflow_reason(self, health):
        overflow = int(health.get("overflow_frames", 0) or 0)
        interval_overflow = int(health.get("interval_overflow_frames", 0) or 0)
        if (
            str(health.get("status")) == "OVERFLOW"
            or interval_overflow > 0
            or overflow > self._record_camera_overflow_baseline
        ):
            first = health.get("first_overflow_frame_index")
            suffix = "" if first is None else f" at camera frame {first}"
            return f"camera delivery FIFO overflow{suffix}; accepted frames were not overwritten"
        if str(health.get("status")) == "ERROR":
            return f"camera delivery FIFO failed: {health.get('error') or 'unknown error'}"
        return None

    def _drain_camera_delivery_locked(self):
        """Move only downstream-capacity-sized, individually acknowledged source frames."""

        if not self._camera_delivery_enabled:
            return 0
        camera = self._cam
        drain = getattr(camera, "drain_delivery", None)
        if not callable(drain):
            self._abort_recording(
                "camera delivery FIFO disappeared during recording",
                boundary_closed=True,
            )
            return 0

        camera_health = self._camera_delivery_health()
        reason = self._camera_delivery_overflow_reason(camera_health)
        if reason is not None:
            self._abort_recording(reason, boundary_closed=True)
            return 0

        ingress_health = self._ingress_worker.health()
        if ingress_health.get("status") == "ERROR":
            self._abort_recording(
                f"diagnostic capture ingress failed: {ingress_health.get('error')}",
                boundary_closed=True,
            )
            return 0
        free_slots = max(
            0,
            int(ingress_health.get("queue_capacity", self.capture_queue_frames))
            - int(ingress_health.get("queue_depth", 0)),
        )
        moved = 0
        # Pop exactly one frame only after proving one downstream slot exists.
        # A popped frame remains the sole pending frame until its exact robot
        # exposure upper bracket arrives; no later camera frame may overtake it.
        for _ in range(free_slots):
            frame = self._pending_delivery_frame
            if frame is None:
                try:
                    batch = drain(max_frames=1)
                except Exception as exc:  # noqa: BLE001
                    self._abort_recording(
                        f"camera delivery FIFO drain failed: {type(exc).__name__}: {exc}",
                        boundary_closed=True,
                    )
                    return moved
                if not batch:
                    break
                if len(batch) != 1:
                    self._abort_recording(
                        "camera delivery FIFO violated one-frame acknowledged drain",
                        boundary_closed=True,
                    )
                    return moved
                frame = batch[0]
                self._pending_delivery_frame = frame
            outcome = self._submit_camera_capture(frame, source="delivery")
            if outcome is None:
                # Robot reception is independent and remains live while this
                # bounded source frame is retried on the next collector pass.
                return moved
            if outcome is not True:
                return moved
            self._pending_delivery_frame = None
            moved += 1

        camera_health = self._camera_delivery_health()
        reason = self._camera_delivery_overflow_reason(camera_health)
        if reason is not None:
            self._abort_recording(reason, boundary_closed=True)
        return moved

    def _drain_camera_delivery(self):
        with self._ingress_submit_lock:
            return self._drain_camera_delivery_locked()

    def _flush_camera_delivery_locked(self, *, timeout_s=5.0):
        """After source End, losslessly hand the fixed camera tail to disk ingress."""

        deadline = time.monotonic() + float(timeout_s)
        while self.platform.state is CollectionState.RECORDING:
            moved = self._drain_camera_delivery_locked()
            if self.platform.state is not CollectionState.RECORDING:
                return False
            health = self._camera_delivery_health()
            reason = self._camera_delivery_overflow_reason(health)
            if reason is not None:
                self._abort_recording(reason, boundary_closed=True)
                return False
            if (
                int(health.get("queue_depth", 0) or 0) == 0
                and self._pending_delivery_frame is None
            ):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._abort_recording(
                    "camera delivery FIFO could not hand off its End tail before timeout",
                    boundary_closed=True,
                )
                return False
            ingress = self._ingress_worker.health()
            target = ingress.get("last_accepted_index")
            if self._pending_delivery_frame is not None:
                time.sleep(min(0.002, remaining))
                continue
            if target is None or int(target) < 0:
                if moved == 0:
                    self._abort_recording(
                        "camera delivery FIFO made no downstream handoff progress",
                        boundary_closed=True,
                    )
                    return False
                continue
            if not self._ingress_worker.drain(
                through_index=int(target), timeout=min(remaining, 0.5),
            ):
                ingress = self._ingress_worker.health()
                if ingress.get("status") == "ERROR":
                    self._abort_recording(
                        f"diagnostic capture ingress failed: {ingress.get('error')}",
                        boundary_closed=True,
                    )
                    return False
        return False

    @staticmethod
    def _safe_worker_health(worker, *, label):
        if worker is None:
            return {
                "status": "STOPPED", "thread_alive": False,
                "error": f"{label} worker is unavailable",
            }
        try:
            value = worker.health()
        except Exception as exc:  # health endpoints themselves are watchdog inputs
            return {
                "status": "ERROR", "thread_alive": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(value, dict):
            return {
                "status": "ERROR", "thread_alive": False,
                "error": f"{label} health is not a mapping",
            }
        return dict(value)

    def _runtime_health_snapshot(self, *, now=None):
        """Return the strict, bounded liveness gate shared by UI and Confirm."""

        now = time.monotonic() if now is None else float(now)
        collector_alive = self._thread is not None and self._thread.is_alive()
        ingress = self._safe_worker_health(self._ingress_worker, label="capture ingress")
        preview = self._safe_worker_health(self._preview_worker, label="preview")
        model = (
            self._safe_worker_health(self._model_worker, label="model")
            if self._model_worker is not None else {
                "status": "OK" if self.fake and self._model_status == "RUNNING" else "STOPPED",
                "thread_alive": collector_alive if self.fake else False,
                "last_result_age_s": (
                    0.0 if self.fake and self._model_status == "RUNNING" else None
                ),
            }
        )
        camera = self._camera_delivery_health()
        preview_age = (
            None if self._preview_frame_t is None
            else max(0.0, now - float(self._preview_frame_t))
        )
        try:
            fps = self._negotiated_camera_fps()
        except RuntimeError:
            fps = self.diagnostic_fps
        preview_freshness_limit = max(self.max_tracking_age_s, 3.0 / float(fps))
        blockers = []
        if not collector_alive:
            blockers.append("collector loop thread is not alive")
        if self._loop_error is not None:
            blockers.append(self._loop_error)
        if self._cam is None or self._camera_status != "OK":
            blockers.append(f"camera status is {self._camera_status}")
        # Production CameraStream publishes strict capture-thread evidence.
        # Small injected fake cameras used in tests may not expose it; the
        # collector loop itself is their source and remains mandatory above.
        if not self.fake or "thread_alive" in camera:
            if camera.get("status") != "OK":
                blockers.append(f"camera delivery status is {camera.get('status')}")
            if camera.get("thread_alive") is not True:
                blockers.append("camera capture thread is not alive")
            if camera.get("frame_fresh") is not True:
                blockers.append("camera frame is stale")
        if model.get("status") != "OK" or model.get("thread_alive") is not True:
            blockers.append("model worker is not alive and OK")
        model_age = model.get("last_result_age_s")
        if model_age is None or float(model_age) > self.max_tracking_age_s:
            blockers.append("model output is stale")
        if preview.get("status") != "OK" or preview.get("thread_alive") is not True:
            blockers.append("preview worker is not alive and OK")
        if preview_age is None or preview_age > preview_freshness_limit:
            blockers.append("preview is stale")
        if ingress.get("status") != "OK" or ingress.get("thread_alive") is not True:
            blockers.append("capture ingress worker is not alive and OK")
        if self._fatal_error is not None:
            blockers.append(str(self._fatal_error))
        robot = self._robot_snapshot()
        if self.require_robot and robot.get("ready") is not True:
            blockers.append(
                "left OpenArm telemetry is not ready: "
                + str(robot.get("error") or robot.get("status") or "unknown")
            )
        # P0-4: the Hall-force channel is advisory-only and never gates Start. A missing
        # or degraded force reading does not block recording; it is recorded when present
        # and honestly marked absent/degraded otherwise. The snapshot is still surfaced
        # for the physician's reference, it just contributes no blocker.
        force = self._force_snapshot()
        return {
            "strict_ok": not blockers,
            "blockers": tuple(dict.fromkeys(blockers)),
            "collector_thread_alive": collector_alive,
            "collector_loop_error": self._loop_error,
            "camera": camera,
            "model": model,
            "preview": preview,
            "ingress": ingress,
            "preview_age_s": preview_age,
            "preview_freshness_limit_s": preview_freshness_limit,
            "robot": robot,
            "force": force,
        }

    def _observe_runtime_health(self, *, runtime=None):
        """Advance the health epoch on a healthy->degraded edge; return strict_ok.

        Called every loop iteration.  Edge-triggering means a worker that dies
        and is then restarted still bumps the generation, so a Start in progress
        during that flap is forced to roll back even though liveness has since
        recovered by the final check.
        """

        runtime = self._runtime_health_snapshot() if runtime is None else runtime
        strict = runtime.get("strict_ok") is True
        with self._health_epoch_lock:
            self._advance_health_epoch_locked(strict)
        return strict

    def _advance_health_epoch_locked(self, strict):
        """Advance the epoch on a healthy->degraded edge.  Caller holds the lock."""
        if not strict and self._last_runtime_strict_ok:
            self._health_generation += 1
        self._last_runtime_strict_ok = strict

    def arm_interval_if_healthy(self, expected_generation):
        """Atomically re-check liveness at the exact health epoch and open ingress.

        The health-epoch bump (driven by a worker death observed on the loop
        thread) and this final check-and-open are serialised by ``_health_epoch_lock``
        so no worker death can be observed *between* the check and the ingress open:
        either the epoch has already advanced (and we refuse), or it cannot advance
        until after ingress is open (and the recording's ongoing liveness monitor
        catches any death from that point).  Returns True iff ingress was opened.
        """
        runtime = self._runtime_health_snapshot()
        rejection = self._device_start_rejection(runtime=runtime)
        with self._health_epoch_lock:
            self._advance_health_epoch_locked(runtime.get("strict_ok") is True)
            if rejection is not None or self._health_generation != expected_generation:
                return False
            self._capture_ingress_paused = False
            return True

    def _device_start_rejection(self, *, runtime=None):
        """Collector-owned device *liveness* precondition for Start.

        This is the exact strict, bounded liveness gate Confirm already enforces
        (_runtime_health_snapshot): the collector loop, camera capture thread,
        model/preview/ingress workers must all be alive AND fresh, and the
        required Hall force and OpenArm robot ready.  Cached ``camera_status``/
        ``model_status`` strings are NOT trusted — a dead thread leaves them
        stale — so this catches a model worker that died, a 99-second-stale Hall
        reading, or a camera thread that stopped delivering.  The served
        snapshot's start_authorized and the control("start") path both consult
        it, so the page and the API can never diverge, and it is re-checked
        immediately before the arm to shrink the confirm->arm TOCTOU window.
        """

        runtime = self._runtime_health_snapshot() if runtime is None else runtime
        if runtime.get("strict_ok") is not True:
            blockers = runtime.get("blockers") or ("runtime is not healthy",)
            return "device/runtime is not ready: " + "; ".join(
                str(item) for item in blockers
            )
        return None

    def _control_locked(
        self, action, *, preview_frame_index=None, preview_frame_token=None,
    ):
        normalized = action.strip().lower() if isinstance(action, str) else ""
        if self.commissioning_preview_only and normalized in {
            "start", "end", "confirm", "redo", "prev", "next",
        }:
            return {
                "ok": False,
                "code": "commissioning_preview_only",
                "message": "Recording is disabled in commissioning preview",
                "state": self.platform.state.value,
            }
        # Left-arm controller lifecycle from the console (real motion).  Disable is
        # always honoured; bring-up is gated to a configured, non-preview collector.
        if normalized == "arm_disable":
            return self.disable_controller()
        if normalized == "arm_bringup":
            return self.bring_up_controller()
        if normalized == "confirm_back":
            token_valid = (
                type(preview_frame_index) is int
                and self._preview_frame_token_valid(
                    preview_frame_index, preview_frame_token,
                )
            )
            if not token_valid:
                return {
                    "ok": False,
                    "code": "preview_frame_not_bound",
                    "message": "Confirm Back requires the displayed camera frame",
                    "state": self.platform.state.value,
                    "back_confirmation": self._back_confirmation_snapshot(),
                }
            gate = self._back_gate
            confirm = getattr(gate, "confirm_back", None) if gate is not None else None
            if not callable(confirm):
                return {
                    "ok": False,
                    "code": "back_gate_not_ready",
                    "message": "Person detector is still starting",
                    "state": self.platform.state.value,
                    "back_confirmation": self._back_confirmation_snapshot(),
                }
            try:
                result = dict(confirm(preview_frame_index=preview_frame_index))
            except Exception as exc:  # noqa: BLE001 - never claim a confirmation on error
                self._clear_back_confirmation("BACK_CONFIRMATION_ERROR")
                return {
                    "ok": False,
                    "code": "back_confirmation_error",
                    "message": f"Back confirmation failed: {type(exc).__name__}: {exc}",
                    "state": self.platform.state.value,
                    "back_confirmation": self._back_confirmation_snapshot(),
                }
            result["state"] = self.platform.state.value
            result.setdefault("back_confirmation", self._back_confirmation_snapshot())
            return result
        start_back_gate = None
        start_back_token = None
        if normalized == "start":
            # Single Start authority, shared with the served snapshot: the
            # platform decision (mode/preflight/writer/state) AND the
            # collector-owned device readiness (camera/model up, no fatal error).
            # Device readiness is exactly the piece the platform cannot see and
            # the one the reviewer saw the page block while the API allowed; the
            # specific audio/back gates below then add their own precise codes.
            start_rejection = self.platform.start_authorization()
            if start_rejection is None:
                start_rejection = self._device_start_rejection()
            if start_rejection is not None:
                return {
                    "ok": False,
                    "code": "start_not_authorized",
                    "message": start_rejection,
                    "state": self.platform.state.value,
                }
            back_confirmation = self._back_confirmation_snapshot()
            if (
                back_confirmation.get("required") is True
                and back_confirmation.get("confirmed") is not True
            ):
                return {
                    "ok": False,
                    "code": "back_confirmation_required",
                    "message": "Face away, stay alone in frame, then Confirm Back",
                    "state": self.platform.state.value,
                    "back_confirmation": back_confirmation,
                }
            if back_confirmation.get("required") is True:
                start_back_gate = self._back_gate
                begin = getattr(start_back_gate, "begin_start_confirmation", None)
                if not callable(begin):
                    self._clear_back_confirmation("START_FAILED")
                    return {
                        "ok": False,
                        "code": "back_confirmation_required",
                        "message": "Back confirmation cannot authorize Start",
                        "state": self.platform.state.value,
                    }
                try:
                    start_back_token, back_confirmation = begin()
                except BackConfirmationError as exc:
                    self._clear_back_confirmation("START_FAILED")
                    return {
                        "ok": False,
                        "code": "back_confirmation_required",
                        "message": str(exc),
                        "state": self.platform.state.value,
                    }
        if normalized == "start" and (self.require_audio or self.audio_device is not None):
            audio = self._audio_snapshot()
            if audio["status"] != "OK":
                self._clear_back_confirmation("START_FAILED")
                return {
                    "ok": False,
                    "code": "audio_not_ready",
                    "message": (
                        "Start blocked because the stethoscope audio stream is not live: "
                        f"{audio['status']}"
                    ),
                    "state": self.platform.state.value,
                }
            if self.require_audio and audio.get("signal_present") is not True:
                self._clear_back_confirmation("START_FAILED")
                return {
                    "ok": False,
                    "code": "audio_signal_missing",
                    "message": "Start blocked because no stethoscope signal is present",
                    "state": self.platform.state.value,
                }
        if normalized == "confirm" and self.platform.state is CollectionState.RECORDING:
            runtime = self._runtime_health_snapshot()
            if runtime["strict_ok"] is not True:
                return {
                    "ok": False,
                    "code": "runtime_not_ready",
                    "message": "Confirm blocked because the live collection pipeline is not healthy",
                    "state": self.platform.state.value,
                    "runtime_blockers": list(runtime["blockers"]),
                }
            with self._lock:
                safe = bool(self._current_tracking.get("constellation_valid", False))
                reason = str(self._current_tracking.get("reason", "back not tracked"))
            if not safe:
                return {
                    "ok": False,
                    "code": "back_not_tracked",
                    "message": f"stable 10-point posterior constellation is unavailable: {reason}",
                    "state": self.platform.state.value,
                }
        if normalized == "start":
            # Initialize every durable output first. Only then atomically arm
            # CameraStream; slow file creation is deliberately outside the
            # recording interval and cannot consume its first frames.
            with self._ingress_submit_lock:
                # TOCTOU re-check: device/worker liveness may have dropped during
                # back confirmation or audio preflight since the top-of-Start
                # gate.  Re-check immediately before opening the interval so a
                # worker that just died can never arm an empty recording.
                arm_rejection = self._device_start_rejection()
                if arm_rejection is not None:
                    self._clear_back_confirmation("START_FAILED")
                    return {
                        "ok": False,
                        "code": "start_not_authorized",
                        "message": arm_rejection,
                        "state": self.platform.state.value,
                    }
                # Duration-aware storage gate: never open a recording that cannot
                # hold the planned session length plus the finalization reserve
                # (a static min-free watermark would still let Start proceed with
                # only a second of headroom, then fail mid-record).
                storage_rejection = self._storage_start_rejection(
                    self.planned_session_duration_s,
                )
                if storage_rejection is not None:
                    self._clear_back_confirmation("START_FAILED")
                    return {
                        "ok": False,
                        "code": "insufficient_storage_for_planned_session",
                        "message": storage_rejection,
                        "state": self.platform.state.value,
                    }
                # Snapshot the health epoch here, at the last known-healthy point,
                # so the final arm can prove nothing degraded while the session,
                # capture, and sensors were being brought up.
                health_gen_before = self._health_generation
                if not self._ingress_worker.begin_interval():
                    self._clear_back_confirmation("START_FAILED")
                    return {
                        "ok": False,
                        "code": "capture_ingress_not_ready",
                        "message": "capture FIFO could not open a clean recording interval",
                        "state": self.platform.state.value,
                    }
                self._capture_ingress_paused = True
                self._recording_boundary = None
                self._camera_delivery_enabled = False
                self._camera_delivery_interval_id = None
                self._pending_delivery_frame = None
                self._latest_recorded_robot_evidence = None
                self._pending_model_frame = None
                result = self.platform.control(action)
                if result.get("ok"):
                    if self.require_force:
                        # P0-4: force is an advisory reference channel, not a collection
                        # gate. Arm it opportunistically; if it cannot arm, record
                        # WITHOUT it (honestly marked degraded) rather than failing Start.
                        try:
                            session_dir = self.platform.session_dir
                            if session_dir is None:
                                raise RuntimeError(
                                    "Hall-force capture has no allocated sensor session"
                                )
                            if self._force is None:
                                raise RuntimeError("live Hall-force owner is unavailable")
                            self._force.prepare_recording(session_dir)
                            self._force_recording_active = True
                        except Exception as exc:  # noqa: BLE001 - advisory, never blocks
                            self._force_recording_active = False
                            self._force_status = "ERROR"
                            self._force_error = (
                                "Hall-force channel unavailable at Start (advisory, "
                                f"recording without it): {type(exc).__name__}: {exc}"
                            )
                    robot_camera_arm = None
                    try:
                        session_dir = self.platform.session_dir
                        if session_dir is None:
                            raise RuntimeError(
                                "robot capture has no allocated sensor session"
                            )
                        self._occlusion_episode_id = (
                            "session-" + Path(session_dir).name
                        )
                        robot_camera_arm = self._start_robot_session(session_dir)
                    except Exception as exc:  # noqa: BLE001 - camera remains unarmed
                        reason = (
                            "left OpenArm Start preflight failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._robot_status = "ERROR"
                        self._robot_error = reason
                        return self._fail_start_after_platform(
                            reason, code="robot_start_not_authorized",
                        )
                    # Final liveness arbitration, immediately before latching the
                    # camera delivery boundary (the true arm).  If a worker died
                    # at any point during session creation, capture.start(), or
                    # Hall/robot init — even a flap that has since recovered — the
                    # health epoch has advanced or liveness is currently degraded,
                    # so roll the entire Start back rather than arm an empty or
                    # gap-corrupted recording.
                    if (
                        self._device_start_rejection() is not None
                        or self._health_generation != health_gen_before
                    ):
                        return self._fail_start_after_platform(
                            "runtime health degraded during Start arm",
                            code="runtime_degraded_during_start",
                        )
                    boundary_camera_index = int(self._last_camera_index)
                    boundary_frame_t = None
                    start_t = time.monotonic()
                    delivery_proof = None
                    begin_delivery = getattr(
                        self._cam, "begin_delivery_interval", None,
                    )
                    try:
                        if callable(begin_delivery):
                            start_boundary = begin_delivery(timeout_s=2.0)
                            start_t = float(getattr(start_boundary, "t_start"))
                            raw_proof = getattr(start_boundary, "proof", None)
                            if not isinstance(raw_proof, dict):
                                raise RuntimeError(
                                    "camera Start boundary returned no durability proof"
                                )
                            delivery_proof = dict(raw_proof)
                            if delivery_proof.get("boundary_latched_atomically") is not True:
                                raise RuntimeError(
                                    "camera Start boundary was not latched atomically"
                                )
                            boundary_frame = getattr(start_boundary, "frame", None)
                            proof_index = delivery_proof.get(
                                "start_boundary_frame_index", -1,
                            )
                            boundary_camera_index = int(
                                getattr(boundary_frame, "index", proof_index)
                            )
                            boundary_frame_t = delivery_proof.get(
                                "start_boundary_host_t",
                                getattr(boundary_frame, "t", None),
                            )
                            interval_id = getattr(start_boundary, "interval_id", None)
                            if interval_id is None:
                                interval_id = delivery_proof.get("interval_id")
                            self._camera_delivery_interval_id = (
                                None if interval_id is None else int(interval_id)
                            )
                            self._camera_delivery_enabled = True
                        elif self._cam is not None:
                            # Compatibility for injected synthetic test cameras.
                            # Production CameraStream always exposes the FIFO API.
                            if not self.fake:
                                raise RuntimeError(
                                    "production camera lacks lossless delivery FIFO"
                                )
                            boundary_frame = self._cam.latest_frame()
                            if (
                                getattr(boundary_frame, "status", None) == "OK"
                                and getattr(boundary_frame, "frame", None) is not None
                            ):
                                boundary_camera_index = int(boundary_frame.index)
                                boundary_frame_t = float(boundary_frame.t)
                        if not math.isfinite(start_t):
                            raise RuntimeError("camera Start boundary timestamp is invalid")
                    except Exception as exc:  # noqa: BLE001 - fail closed after file init
                        reason = (
                            "camera delivery Start boundary failed after file initialization: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        return self._fail_start_after_platform(
                            reason, code="camera_delivery_not_ready",
                        )
                    self._recording_boundary = {
                        "start_host_t_s": start_t,
                        "camera_start_boundary_index": boundary_camera_index,
                        "camera_start_boundary_host_t_s": boundary_frame_t,
                        "camera_delivery_interval_id": self._camera_delivery_interval_id,
                        "camera_delivery_start_proof": delivery_proof,
                        "model_start_boundary_index": int(self._last_result_index),
                    }
                    if robot_camera_arm is not None:
                        self._recording_boundary["robot_camera_arm"] = (
                            robot_camera_arm
                        )
                    if self.require_force:
                        try:
                            self._force.begin_recording(t0=start_t)
                            self._recording_boundary["force_start_boundary"] = {
                                "host_t_s": start_t,
                                "hardware": copy.deepcopy(self._force_hardware),
                                "calibration_sha256": self.hall_calibration_sha256,
                                "sync_calibration_sha256": self.sync_calibration_sha256,
                                "sync": copy.deepcopy(self._force_sync_metadata),
                            }
                        except Exception as exc:  # noqa: BLE001 - advisory, never blocks
                            # P0-4: force is advisory. If it cannot latch its Start
                            # boundary, record WITHOUT it (marked degraded) rather than
                            # failing Start.
                            self._force_recording_active = False
                            self._force_status = "ERROR"
                            self._force_error = (
                                "Hall-force Start boundary unavailable (advisory, "
                                f"recording without it): {type(exc).__name__}: {exc}"
                            )
                    if self._audio is not None:
                        try:
                            session_dir = self.platform.session_dir
                            if session_dir is None:
                                raise RuntimeError("audio capture has no allocated session")
                            self._audio.begin_recording(session_dir, t0=start_t)
                            self._audio_recording_active = True
                            self._recording_boundary["audio_start_boundary"] = {
                                "host_t_s": start_t,
                                "device": self.audio_device,
                                "input_sample_rate": self.audio_sample_rate,
                                "model_sample_rate": 16_000,
                                "channels": self.audio_channels,
                                "pick_channel": self.audio_pick_channel,
                                "hardware": copy.deepcopy(self._audio_hardware),
                            }
                        except Exception as exc:  # noqa: BLE001
                            reason = (
                                "stethoscope audio Start boundary failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            return self._fail_start_after_platform(
                                reason, code="audio_start_not_latched",
                            )
                    setter = getattr(
                        self.platform.capture, "set_recording_boundary", None,
                    )

                    def persist_start_boundary(snapshot):
                        self._recording_boundary["back_confirmation_at_start"] = dict(
                            snapshot,
                        )
                        if callable(setter):
                            setter(self._recording_boundary)

                    def arm_start(_snapshot):
                        # FINAL liveness gate, synchronously re-evaluated the instant
                        # before capture ingress opens.  The earlier arm-time check
                        # (before the camera boundary latch) leaves a window up to
                        # here; a worker death anywhere in that window must still roll
                        # the Start back rather than open ingress on a dead model.
                        # Any raise here is caught below and fully rolled back.
                        self._end_boundary_closing = False
                        self._completion_proof_cache = None
                        ingress_health = self._ingress_worker.health()
                        self._record_ingress_drop_baseline = int(
                            ingress_health.get("dropped_before_inference", 0)
                        )
                        self._record_ingress_overflow_baseline = int(
                            ingress_health.get("overflow_rejections", 0)
                        )
                        camera_health = self._camera_delivery_health()
                        self._record_camera_overflow_baseline = int(
                            camera_health.get("overflow_frames", 0) or 0
                        )
                        self._record_last_camera_index = None
                        # Atomic: re-check the health epoch and open capture ingress in
                        # one step, so a worker death cannot be observed between them.
                        if not self.arm_interval_if_healthy(health_gen_before):
                            raise RuntimeError(
                                "runtime health degraded before capture ingress open"
                            )

                    try:
                        if start_back_gate is None:
                            snapshot = self._back_confirmation_snapshot()
                            persist_start_boundary(snapshot)
                            arm_start(snapshot)
                        else:
                            commit = getattr(
                                start_back_gate, "commit_start_confirmation", None,
                            )
                            if not callable(commit):
                                raise BackConfirmationError(
                                    "Back confirmation cannot commit Start"
                                )
                            commit(
                                start_back_token,
                                prepare=persist_start_boundary,
                                arm=arm_start,
                            )
                    except BackConfirmationError as exc:
                        reason = f"back confirmation lost during Start: {exc}"
                        return self._fail_start_after_platform(
                            reason,
                            code="back_confirmation_lost_during_start",
                        )
                    except Exception as exc:
                        reason = (
                            "capture Start boundary could not be persisted and armed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        return self._fail_start_after_platform(
                            reason,
                            code="capture_start_boundary_not_persisted",
                        )
                else:
                    self._ingress_worker.close_interval()
                    self._recording_boundary = None
                    self._clear_back_confirmation("START_FAILED")
        elif normalized in {"end", "stop"}:
            # First close CameraStream against its producer condition, then hand
            # every <=End source frame to the disk FIFO, and only then close the
            # downstream interval. This prevents a decoded tail frame from
            # arriving after an apparently successful finalization.
            target = None
            camera_end_proof = None
            camera_end_t = None
            camera_delivery_health = self._camera_delivery_health()
            if self.platform.state is CollectionState.RECORDING:
                with self._ingress_submit_lock:
                    self._end_boundary_closing = True
                    if self._camera_delivery_enabled:
                        latch = getattr(self._cam, "latch_end_boundary", None)
                        if not callable(latch):
                            reason = "camera delivery FIFO lacks atomic End handoff"
                            self._abort_recording(reason, boundary_closed=True)
                            return {
                                "ok": False, "code": "camera_end_not_latched",
                                "message": reason, "state": self.platform.state.value,
                            }
                        try:
                            camera_end = latch(
                                timeout_s=2.0,
                                interval_id=self._camera_delivery_interval_id,
                            )
                            camera_end_t = float(getattr(camera_end, "t_end"))
                            raw_proof = getattr(camera_end, "proof", None)
                            if not isinstance(raw_proof, dict):
                                raise RuntimeError(
                                    "camera End boundary returned no durability proof"
                                )
                            camera_end_proof = dict(raw_proof)
                            if camera_end_proof.get("boundary_latched_atomically") is not True:
                                raise RuntimeError(
                                    "camera End boundary was not latched atomically"
                                )
                            if not math.isfinite(camera_end_t):
                                raise RuntimeError(
                                    "camera End boundary timestamp is invalid"
                                )
                        except Exception as exc:  # noqa: BLE001
                            reason = (
                                "camera End boundary handoff failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            self._abort_recording(reason, boundary_closed=True)
                            return {
                                "ok": False, "code": "camera_end_not_latched",
                                "message": reason, "state": self.platform.state.value,
                            }
                        if not self._flush_camera_delivery_locked(timeout_s=5.0):
                            reason = self.platform.last_error or (
                                "camera delivery FIFO End tail could not be drained"
                            )
                            return {
                                "ok": False, "code": "camera_delivery_not_drained",
                                "message": reason, "state": self.platform.state.value,
                            }
                        camera_delivery_health = self._camera_delivery_health()
                    self._capture_ingress_paused = True
                    target = self._ingress_worker.close_interval()
            if self.platform.state is CollectionState.RECORDING:
                drained = self._ingress_worker.drain(
                    through_index=target, timeout=5.0,
                )
                ingress_health = self._ingress_worker.health()
                dropped = (
                    int(ingress_health.get("dropped_before_inference", 0))
                    - self._record_ingress_drop_baseline
                )
                overflowed = (
                    int(ingress_health.get("overflow_rejections", 0))
                    - self._record_ingress_overflow_baseline
                )
                if (
                    not drained
                    or ingress_health.get("status") == "ERROR"
                    or dropped > 0
                    or overflowed > 0
                    or (
                        target is not None
                        and int(ingress_health.get("last_completed_index", -1)) != int(target)
                    )
                ):
                    reason = (
                        "capture ingress failed to drain before End"
                        if not drained else
                        (
                            f"capture ingress failed before End: {ingress_health.get('error')}"
                            if ingress_health.get("status") == "ERROR" else
                            (
                                f"capture ingress dropped {dropped} frame(s) before End"
                                if dropped > 0 else
                                (
                                    "capture ingress overflowed before End"
                                    if overflowed > 0
                                    else "capture ingress did not complete the exact End frame"
                                )
                            )
                        )
                    )
                    self._abort_recording(reason, boundary_closed=True)
                    return {
                        "ok": False,
                        "code": "capture_ingress_not_drained",
                        "message": reason,
                        "state": self.platform.state.value,
                    }
                end_boundary = {
                    **(self._recording_boundary or {}),
                    "end_host_t_s": camera_end_t,
                    "camera_end_boundary_proof": camera_end_proof,
                    "camera_delivery_fifo": camera_delivery_health,
                    "capture_ingress_fifo": ingress_health,
                    "last_submitted_camera_index": target,
                    "last_completed_camera_index": ingress_health.get(
                        "last_completed_index",
                    ),
                    "last_completed_model_index": int(self._last_result_index),
                    "end_camera_frame_has_completed_model_result": bool(
                        target is not None and self._last_result_index == int(target)
                    ),
                    "inference_timeline_policy": (
                        "latest_only_recomputable_from_pinned_rgbd"
                    ),
                    "offline_model_recompute_from_pinned_rgbd": True,
                }
                if self._force_recording_active:
                    try:
                        if self._force is None:
                            raise RuntimeError("live Hall-force owner disappeared")
                        force_end_t = (
                            float(camera_end_t)
                            if camera_end_t is not None else time.monotonic()
                        )
                        # P0-4: force is advisory. Record whatever the Hall channel
                        # captured -- INCLUDING over-threshold readings the physician
                        # wants -- with its true status, and bind it if the writer
                        # accepts it, but NEVER abort End: a degraded/missing force
                        # channel does not lose the session or its promotion eligibility.
                        summary = self._force.end_recording(t_end=force_end_t)
                        self._force_recording_active = False
                        force_summary = {
                            "path": "force.csv",
                            "t_start": float(summary.t_start),
                            "t_end": float(summary.t_end),
                            "n_samples": int(summary.n_samples),
                            "status": str(summary.status),
                            "sha256": str(summary.sha256),
                            "device_t_start_ms": float(summary.device_t_start_ms),
                            "device_t_end_ms": float(summary.device_t_end_ms),
                            "calibration_sha256": str(summary.calibration_sha256),
                            "device_identity_sha256": str(
                                summary.device_identity_sha256
                            ),
                        }
                        force_setter = getattr(
                            self.platform.capture, "set_force_summary", None,
                        )
                        if callable(force_setter):
                            force_setter(force_summary)
                        end_boundary["force"] = force_summary
                        end_boundary["force_sync"] = {
                            "calibration_sha256": self.hall_calibration_sha256,
                            "sync_calibration_sha256": self.sync_calibration_sha256,
                            "hardware": copy.deepcopy(self._force_hardware),
                            "sync": copy.deepcopy(self._force_sync_metadata),
                        }
                        self._force_last_summary = dict(force_summary)
                        self._force_status = str(summary.status)
                        self._force_error = (
                            None if str(summary.status) == "OK"
                            else f"Hall-force {summary.status} (advisory)"
                        )
                    except Exception as exc:  # noqa: BLE001 - advisory, never aborts End
                        self._force_recording_active = False
                        self._force_status = "ERROR"
                        self._force_error = (
                            "Hall-force End readout unavailable (advisory): "
                            f"{type(exc).__name__}: {exc}"
                        )
                if self.require_robot:
                    try:
                        robot_summary = self._finish_robot_session(
                            expected_frame_count=int(
                                ingress_health.get("accepted", -1)
                            ),
                        )
                        end_boundary["robot"] = copy.deepcopy(robot_summary)
                    except Exception as exc:  # noqa: BLE001
                        self._robot_status = "ERROR"
                        self._robot_error = f"{type(exc).__name__}: {exc}"
                        reason = (
                            "left OpenArm End verification failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._abort_recording(reason, boundary_closed=True)
                        return {
                            "ok": False,
                            "code": "robot_end_not_verified",
                            "message": reason,
                            "state": self.platform.state.value,
                        }
                if self._audio_recording_active:
                    try:
                        if self._audio is None:
                            raise RuntimeError("live audio owner disappeared")
                        audio_end_t = (
                            float(camera_end_t)
                            if camera_end_t is not None else time.monotonic()
                        )
                        summary = self._audio.end_recording(t_end=audio_end_t)
                        self._audio_recording_active = False
                        if str(summary.status) != "OK":
                            raise RuntimeError(
                                f"audio boundary coverage is {summary.status}"
                            )
                        audio_summary = {
                            "path": "audio.wav",
                            "sample_rate": int(summary.sr),
                            "input_sample_rate": self.audio_sample_rate,
                            "n_samples": int(summary.n_samples),
                            "t_start": float(summary.t_start),
                            "t_end": float(summary.t_end),
                            "status": str(summary.status),
                            "peak_db": float(summary.peak_db),
                            "sha256": str(summary.sha256),
                        }
                        self._audio.cleanup_recording_staging()
                        audio_setter = getattr(
                            self.platform.capture, "set_audio_summary", None,
                        )
                        if not callable(audio_setter):
                            raise RuntimeError("capture cannot bind finalized audio")
                        audio_setter(audio_summary)
                        end_boundary["audio"] = audio_summary
                        self._audio_last_summary = dict(audio_summary)
                    except Exception as exc:  # noqa: BLE001
                        self._audio_recording_active = False
                        reason = (
                            "stethoscope audio End verification failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._abort_recording(reason, boundary_closed=True)
                        return {
                            "ok": False,
                            "code": "audio_end_not_verified",
                            "message": reason,
                            "state": self.platform.state.value,
                        }
                end_model_health = (
                    self._model_worker.health()
                    if self._model_worker is not None
                    else {}
                )
                end_boundary.update({
                    "model_submitted_frames": end_model_health.get("submitted"),
                    "model_processed_frames": end_model_health.get("processed"),
                    "model_dropped_before_inference": end_model_health.get(
                        "dropped_before_inference",
                    ),
                    "model_worker_last_completed_index": end_model_health.get(
                        "last_completed_index",
                    ),
                })
                if camera_end_proof is not None:
                    try:
                        end_boundary["index_closure"] = validate_capture_boundary_closure(
                            end_boundary,
                            n_rgb_frames=int(ingress_health.get("accepted", -1)),
                            expected_fps=self._negotiated_camera_fps(),
                        )
                    except Exception as exc:
                        reason = (
                            "capture End boundary did not close exactly: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._abort_recording(reason, boundary_closed=True)
                        return {
                            "ok": False,
                            "code": "capture_boundary_not_closed",
                            "message": reason,
                            "state": self.platform.state.value,
                        }
                setter = getattr(self.platform.capture, "set_finalization_context", None)
                if callable(setter):
                    try:
                        setter(end_boundary)
                    except Exception as exc:
                        reason = (
                            "capture finalization boundary could not be persisted: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._abort_recording(reason, boundary_closed=True)
                        return {
                            "ok": False,
                            "code": "capture_boundary_not_persisted",
                            "message": reason,
                            "state": self.platform.state.value,
                        }
            result = self.platform.control(action)
            self._end_boundary_closing = False
            if result.get("ok"):
                self._camera_delivery_enabled = False
                self._camera_delivery_interval_id = None
        else:
            result = self.platform.control(action)
        now = time.monotonic()
        if result.get("ok") and normalized == "start":
            self._record_started, self._record_stopped = now, None
        elif result.get("ok") and normalized in {"end", "stop"} and result.get("accepted"):
            self._record_stopped = now
            self._occlusion_episode_id = "preview-" + os.urandom(16).hex()
            self._clear_back_confirmation("BACK_RECONFIRMATION_REQUIRED")
        return result

    def _abort_recording(self, reason, *, boundary_closed=False):
        self._clear_back_confirmation("RUNTIME_FAULT")
        # Once End has closed and drained the camera interval, FINALIZING owns an
        # immutable disk snapshot. Later live camera/model faults must not race
        # capture.finalize() or destroy a long completed acquisition.
        active = self.platform.state is CollectionState.RECORDING or (
            self.platform.state is CollectionState.ERROR
            and not self.platform._end_requested
        )
        if not boundary_closed:
            with self._ingress_submit_lock:
                # End has already closed the exact camera interval and is only
                # draining accepted disk work. A later live model/camera fault
                # belongs to the continuing preview pipeline, not to that sealed
                # interval, and must not turn a long End request into ABORTED.
                if self._end_boundary_closing:
                    return False
                self._capture_ingress_paused = True
                self._ingress_worker.close_interval()
                self._abort_camera_delivery()
        else:
            self._end_boundary_closing = False
            self._abort_camera_delivery()
        if active:
            self._abort_robot_session(reason)
            if self._force_recording_active and self._force is not None:
                try:
                    self._force.abort_recording()
                except Exception:
                    pass
                finally:
                    self._force_recording_active = False
            if self._audio_recording_active and self._audio is not None:
                try:
                    self._audio.abort_recording()
                except Exception:
                    pass
                finally:
                    self._audio_recording_active = False
            self.platform.abort(str(reason))
            self._occlusion_episode_id = "preview-" + os.urandom(16).hex()
        if active and self._record_stopped is None:
            self._record_stopped = time.monotonic()
        return active

    def _set_diagnostic_blockers(self, blockers):
        normalized = tuple(str(item) for item in blockers if str(item))
        if normalized == self._diagnostic_blockers:
            return
        self._diagnostic_blockers = normalized
        self.platform.set_diagnostic_blockers(normalized)

    def _open_camera(self):
        if self._camera_factory is None:
            from ausc.perception.camera_stream import CameraStream
            factory = CameraStream
        else:
            factory = self._camera_factory
        camera_options = {
            "delivery_capacity_frames": self.camera_delivery_frames,
        }
        if not self.fake:
            camera_options.update({
                "width": int(self.camera_width),
                "height": int(self.camera_height),
                "fps": int(self.camera_fps),
                "color_format": str(self.camera_color_format),
                "depth_mode": {
                    "width": int(self.depth_width),
                    "height": int(self.depth_height),
                    "fps": int(self.depth_fps),
                    "format": str(self.depth_format),
                },
                "align_depth_to_color": True,
            })
        else:
            # Keep the synthetic source's declared mode and generation clock
            # identical.  Browser/CI acceptance may deliberately request a
            # lower rate than Orbbec, and the timeline verifier must see the
            # mode that CameraStream actually produces.
            camera_options.update({
                key: value for key, value in {
                    "width": self.camera_width,
                    "height": self.camera_height,
                    "fps": self.camera_fps,
                }.items() if value is not None
            })
        self._cam = open_camera(
            factory,
            backend=self.camera_backend,
            index=self.camera_index,
            camera_id=self.camera_id,
            **camera_options,
        )
        self._camera_status = "WAITING_FOR_FRAME"
        self._fatal_error = None
        self._set_diagnostic_blockers(["camera/model starting"])

    def _attempt_camera_open(self):
        now = time.monotonic()
        if self._cam is not None or now < self._next_camera_open_t:
            return self._cam is not None
        try:
            self._open_camera()
        except Exception as exc:  # never substitute fake frames for real hardware
            self._fatal_error = f"camera open failed: {type(exc).__name__}: {exc}"
            self._camera_status = "RECONNECTING"
            self._model_status = "RECONNECTING"
            self._set_diagnostic_blockers([self._fatal_error])
            self._next_camera_open_t = now + self.camera_reopen_interval_s
            return False
        self._next_camera_open_t = 0.0
        return True

    def _reset_camera_pipeline(self, reason):
        reason = str(reason)
        self._abort_recording(reason)
        self._occlusion_episode_id = "preview-" + os.urandom(16).hex()
        self._clear_back_confirmation("CAMERA_PIPELINE_RESET")
        self._back_gate = None
        worker, self._model_worker = self._model_worker, None
        model_exited = worker is None
        if worker is not None:
            try:
                model_exited = bool(worker.stop())
            except Exception:
                model_exited = False
        preview_worker, self._preview_worker = self._preview_worker, None
        preview_exited = preview_worker is None
        if preview_worker is not None:
            try:
                preview_exited = bool(preview_worker.stop())
            except Exception:
                preview_exited = False
        camera, self._cam = self._cam, None
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        self._frame_ring.clear()
        self._pending_model_frame = None
        self._pending_model_preview = None
        self._latest_recorded_robot_evidence = None
        self._last_camera_index = -1
        self._last_result_index = -1
        if model_exited and preview_exited:
            replacement_preview_worker = LatestFrameWorker(
                self._render_preview_bundle, name="wam-diagnostic-preview",
            ).start()
        else:
            # A predecessor worker thread refused to exit within its stop()
            # timeout (e.g. a hung GPU forward pass).  NEVER spawn a replacement
            # on top of a live thread: two workers would race the single camera
            # stream and silently grow the thread count.  Keep the unresponsive
            # worker visible so health/freshness report the pipeline degraded and
            # every Start rolls back, and mark the fault terminal — a stuck thread
            # cannot be recovered in-process; the unit must restart.
            replacement_preview_worker = preview_worker
            stuck = "model" if not model_exited else "preview"
            reason = f"{reason}; {stuck} worker thread did not exit (restart required)"
        reset_jpeg = encode_jpeg(np.zeros((480, 640, 3), np.uint8))
        with self._lock:
            self._preview_worker = replacement_preview_worker
            self._preview_generation += 1
            self._preview_result_index = -1
            self._preview_frame_index = 0
            self._preview_frame_t = None
            self._preview_frame_confirmable = False
            self._preview_token_key = os.urandom(32)
            self._preview_publish_times.clear()
            self.jpeg = reset_jpeg
        self._latest_tracking = {
            "frame_index": 0,
            "back_present": False,
            "initialized": False,
            "detector_confidence": 0.0,
            "reason": "CAMERA_RECONNECTING",
            "points": [],
        }
        with self._lock:
            self._current_tracking = {
                **self._latest_tracking,
                "constellation_valid": False,
            }
        self.platform.update_tracking(self._current_tracking)
        self._current_depth_plausible = False
        self._fatal_error = reason
        self._camera_status = "RECONNECTING"
        self._model_status = "RECONNECTING"
        self._set_diagnostic_blockers([reason])
        self._next_camera_open_t = time.monotonic() + self.camera_reopen_interval_s

    def _recover_failed_model_worker(self):
        """Reset the real camera/model pipeline after a terminal worker fault."""

        if self.fake or self._model_worker is None:
            return None
        health = self._safe_worker_health(self._model_worker, label="model")
        if (
            health.get("status") != "ERROR"
            and health.get("thread_alive") is not False
        ):
            return None
        detail = str(
            health.get("error")
            or health.get("status")
            or "model worker stopped"
        )
        reason = f"perception model failed; restarting pipeline: {detail}"
        self._reset_camera_pipeline(reason)
        return reason

    def _build_tracker_processor(self, frame):
        if self._processor_factory is not None:
            return self._processor_factory(self, frame)
        verify_file_sha256(self.checkpoint, self.checkpoint_sha256, label="v9 checkpoint")
        verify_file_sha256(self.base_model, self.base_model_sha256, label="DINOv2 base")
        verify_file_sha256(
            self.posterior_gate_checkpoint,
            self.posterior_gate_checkpoint_sha256,
            label="posterior-person gate checkpoint",
        )
        verify_file_sha256(self.runtime_lock, self.runtime_lock_sha256, label="runtime lock")
        from ausc.perception.tracking import (
            Intrinsics, ModelBasedTracker, ModelConfig, NormalsDetector, runtime_frame,
        )

        active = getattr(self._cam, "color_intrinsics", None)
        if not isinstance(active, dict):
            raise RuntimeError("active Orbbec color intrinsics are unavailable")
        try:
            intrinsics = Intrinsics(
                fx=float(active["fx"]), fy=float(active["fy"]),
                cx=float(active["cx"]), cy=float(active["cy"]),
                width=int(active["width"]), height=int(active["height"]),
                depth_scale=float(frame.depth_scale),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("active Orbbec color intrinsics are invalid") from exc
        height, width = frame.frame.shape[:2]
        if (intrinsics.width, intrinsics.height) != (width, height):
            raise RuntimeError(
                "active Orbbec intrinsics do not match RGB frame: "
                f"{intrinsics.width}x{intrinsics.height} vs {width}x{height}"
            )
        if frame.depth is None or tuple(frame.depth.shape) != (height, width):
            raise RuntimeError("v9 runtime requires depth aligned to the active RGB frame")
        detector = NormalsDetector(
            str(self.checkpoint), device="cuda", half=True,
            fill_holes=True, plausibility_gate=True,
        )
        tracker = ModelBasedTracker(
            detector, intrinsics,
            ModelConfig(init_frames=12, min_conf=0.3, coast_max=20),
        )
        person_gate = UltralyticsPersonGate(
            str(self.posterior_gate_checkpoint),
            device=0,
            image_size=640,
            model_release_id=self.posterior_gate_release_id,
            model_sha256=self.posterior_gate_checkpoint_sha256,
        )
        back_gate = OperatorConfirmedBackGate(person_gate)
        self._back_gate = back_gate
        if self.require_robot:
            for provider, label in (
                (self._patient_pose_provider, "metric patient-pose provider"),
                (self._occlusion_runtime, "occlusion safety runtime"),
            ):
                reset = getattr(provider, "reset", None)
                if not callable(reset):
                    raise RuntimeError(f"{label} cannot reset for a new episode")
                try:
                    reset()
                except Exception as exc:
                    raise RuntimeError(
                        f"{label} reset failed before model start"
                    ) from exc
        return TrackerProcessor(
            tracker,
            runtime_frame,
            presence_gate=back_gate,
            require_robot_mask=self.require_robot,
            back_mask_provider=self._back_mask_provider,
            patient_pose_provider=self._patient_pose_provider,
            occlusion_runtime=self._occlusion_runtime,
            episode_id=lambda: self._occlusion_episode_id,
        )

    def _update_provenance(self):
        negotiated = getattr(self._cam, "negotiated_mode", None)
        actual_camera_id = getattr(self._cam, "device_identity", None)
        if not self.fake and actual_camera_id != self.camera_id:
            raise RuntimeError(
                f"Orbbec serial mismatch: requested {self.camera_id!r}, got {actual_camera_id!r}"
            )
        self._provenance = {
            "camera": {
                "backend": self.camera_backend,
                "camera_id": actual_camera_id or self.camera_id or "synthetic",
                "negotiated_mode": negotiated or {"source": "synthetic"},
            },
            "model": {
                "release_id": self.model_release_id,
                "checkpoint_sha256": self.checkpoint_sha256 or "synthetic",
                "base_model_sha256": self.base_model_sha256 or "synthetic",
                "posterior_gate_release_id": self.posterior_gate_release_id,
                "posterior_gate_sha256": (
                    self.posterior_gate_checkpoint_sha256 or "synthetic"
                ),
                "line": (
                    "YOLO11xPosePersonDetector->OperatorBackConfirm->"
                    "NormalsDetector(v9)->ModelBasedTracker"
                ),
            },
            "runtime": {
                "release_id": self.runtime_release_id,
                "lock_sha256": self.runtime_lock_sha256 or "synthetic",
                "code_release_id": self.code_release_id,
                "ffmpeg_path": self.ffmpeg_path or "synthetic",
                "ffmpeg_sha256": self.ffmpeg_sha256 or "synthetic",
            },
        }

    def _fake_tracking(self, frame):
        height, width = frame.frame.shape[:2]
        points = []
        for target in self.targets:
            points.append({
                "name": target.name,
                "pixel": [target.uv[0] * width, target.uv[1] * height],
                "xyz_camera": [0.0, 0.0, 0.9],
                "status": "TRACKING",
                "measured": True,
                "position_std": 0.003,
            })
        return {
            "frame_index": frame.index,
            "host_t": frame.t,
            "device_t": frame.device_t_ms,
            "back_present": True,
            "initialized": True,
            "detector_confidence": 1.0,
            "reason": "TRACKING",
            "points": points,
        }

    def _ingest_bundle(self, bundle):
        self.platform.ingest_frame(
            bundle.camera_frame, bundle.tracking, update_control_state=False,
        )
        return {"frame_index": bundle.index, "back_present": False, "points": []}

    def _submit_ingress(self, frame, tracking):
        bundle = SimpleNamespace(
            index=int(frame.index), t=float(frame.t), camera_frame=frame,
            tracking=tracking,
        )
        failure_reason = None
        with self._ingress_submit_lock:
            if (
                self._capture_ingress_paused
                or self.platform.state is not CollectionState.RECORDING
            ):
                return True
            index = int(frame.index)
            frame_t = float(frame.t)
            boundary = self._recording_boundary
            if boundary is not None and (
                index <= int(boundary["camera_start_boundary_index"])
                or frame_t < float(boundary["start_host_t_s"])
            ):
                # The frame was acquired before Start but reached this lock
                # afterward. It belongs to neither side of the new session.
                return True
            if (
                self._record_last_camera_index is None
                and boundary is not None
                and int(boundary["camera_start_boundary_index"]) >= 0
                and index != int(boundary["camera_start_boundary_index"]) + 1
            ):
                failure_reason = (
                    "camera index gap at first capture frame: expected "
                    f"{int(boundary['camera_start_boundary_index']) + 1}, got {index}"
                )
            elif (
                self._record_last_camera_index is not None
                and index != self._record_last_camera_index + 1
            ):
                failure_reason = (
                    "camera index gap at capture boundary: expected "
                    f"{self._record_last_camera_index + 1}, got {index}"
                )
            elif self._ingress_worker.submit(bundle):
                self._record_last_camera_index = index
                return True
            else:
                health = self._ingress_worker.health()
                if (
                    int(health.get("overflow_rejections", 0))
                    > self._record_ingress_overflow_baseline
                ):
                    failure_reason = "diagnostic capture FIFO overflow"
                elif health.get("error"):
                    failure_reason = (
                        f"diagnostic capture FIFO write failed: {health.get('error')}"
                    )
                else:
                    failure_reason = "diagnostic capture FIFO rejected a camera frame"
            self._capture_ingress_paused = True
            self._ingress_worker.close_interval()
            # Abort while still owning the submission boundary. A concurrent End
            # cannot pass this failure and start finalization first.
            self._abort_recording(failure_reason, boundary_closed=True)
        return False

    def _build_robot_bundle_from_inputs(self, frame, inputs):
        mask_provider = self._robot_mask_provider
        if mask_provider is None:
            raise RuntimeError(
                "required left OpenArm same-frame evidence chain is unavailable"
            )
        exposure = inputs.camera_exposure
        trajectory = inputs.robot_trajectory
        robot_mask = mask_provider.build(
            exposure=exposure,
            trajectory=trajectory,
        )
        evidence = self._frame_evidence_bundle_factory(
            camera_frame=frame,
            camera_id=exposure.camera_id,
            camera_profile_sha256=exposure.camera_profile_sha256,
            alignment_sha256=exposure.alignment_sha256,
            timing_artifact_sha256=exposure.camera_latency_sha256,
            exposure=exposure,
            robot_trajectory=trajectory,
            robot_mask_packet=robot_mask,
            activated_identity=getattr(
                mask_provider, "activated_identity", None,
            ),
        )
        bound_frame = getattr(evidence, "camera_frame", None)
        if (
            bound_frame is None
            or int(getattr(evidence, "index", -1)) != int(frame.index)
            or int(getattr(bound_frame, "index", -1)) != int(frame.index)
            or getattr(evidence, "robot_mask_packet", None) is None
        ):
            raise RuntimeError(
                "same-frame robot evidence builder returned a mismatched frame"
            )
        return evidence

    def _build_recorded_robot_evidence(self, frame):
        coordinator = self._robot_coordinator
        if coordinator is None:
            raise RuntimeError(
                "required left OpenArm same-frame evidence chain is unavailable"
            )
        device_t_ms = getattr(frame, "device_t_ms", None)
        if device_t_ms is None:
            raise RuntimeError(
                "camera frame has no device timestamp for robot synchronization"
            )
        inputs = coordinator.build_frame_evidence_inputs(
            frame_index=int(frame.index),
            camera_host_t_s=float(frame.t),
            camera_device_t_ms=float(device_t_ms),
        )
        evidence = self._build_robot_bundle_from_inputs(frame, inputs)
        return evidence, dict(inputs.persisted_index)

    def _model_input_for_frame(self, frame):
        if not self.require_robot:
            return frame
        if self.platform.state is CollectionState.RECORDING:
            evidence = self._latest_recorded_robot_evidence
            if (
                evidence is None
                or int(getattr(evidence, "index", -1)) != int(frame.index)
            ):
                raise RobotCaptureNotReady(
                    "persisted same-frame robot evidence is not ready for the model"
                )
            return evidence
        coordinator = self._robot_coordinator
        if coordinator is None:
            raise RobotCaptureNotReady(
                "left OpenArm live evidence coordinator is unavailable"
            )
        device_t_ms = getattr(frame, "device_t_ms", None)
        if device_t_ms is None:
            raise RobotCaptureCoordinatorError(
                "camera frame has no device timestamp for robot synchronization"
            )
        inputs = coordinator.build_live_frame_evidence_inputs(
            frame_index=int(frame.index),
            camera_host_t_s=float(frame.t),
            camera_device_t_ms=float(device_t_ms),
        )
        return self._build_robot_bundle_from_inputs(frame, inputs)

    @staticmethod
    def _bind_robot_tracking_metadata(tracking, evidence, persisted_index):
        result = dict(tracking)
        robot_mask = evidence.robot_mask_packet
        canonical_builder = getattr(evidence, "canonical_metadata_bytes", None)
        if not callable(canonical_builder):
            raise RuntimeError(
                "same-frame robot evidence lacks replayable canonical metadata"
            )
        canonical_bytes = canonical_builder()
        if not isinstance(canonical_bytes, bytes) or not canonical_bytes:
            raise RuntimeError(
                "same-frame robot evidence canonical metadata is invalid"
            )
        try:
            canonical_value = json.loads(canonical_bytes.decode("ascii"))
            reencoded = json.dumps(
                canonical_value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "same-frame robot evidence canonical metadata is invalid"
            ) from exc
        if (
            type(canonical_value) is not dict
            or canonical_value.get("schema") != "wam_frame_evidence_bundle_v3"
            or reencoded != canonical_bytes
        ):
            raise RuntimeError(
                "same-frame robot evidence metadata is not canonical bundle v3"
            )
        canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        if str(evidence.bundle_id) != "frame_evidence-" + canonical_sha256:
            raise RuntimeError(
                "same-frame robot evidence bundle ID does not match canonical metadata"
            )
        result.update({
            "frame_evidence_bundle_id": str(evidence.bundle_id),
            "frame_evidence": {
                "bundle_id": str(evidence.bundle_id),
                "canonical_json": canonical_bytes.decode("ascii"),
                "canonical_sha256": canonical_sha256,
                "schema": "wam_frame_evidence_ledger_record_v1",
            },
            "robot_pair": "left:can1->can3",
            "robot_exact_control_trajectory_sha256": str(
                evidence.robot_trajectory_sha256
            ),
            "robot_exact_action_sha256": str(
                evidence.robot_exact_action_sha256
            ),
            "robot_exact_action_tick_first": int(
                evidence.robot_exact_action_tick_first
            ),
            "robot_exact_action_tick_last": int(
                evidence.robot_exact_action_tick_last
            ),
            "robot_exact_action_count": int(
                evidence.robot_exact_action_count
            ),
            "robot_mask": {
                "frame_index": int(robot_mask.frame_index),
                "mask_sha256": str(robot_mask.mask_sha256),
                "geometry_release_id": str(robot_mask.geometry_release_id),
                "tool_release_id": str(robot_mask.tool_release_id),
                "handeye_release_id": str(robot_mask.handeye_release_id),
                "handeye_sha256": str(robot_mask.handeye_sha256),
                "handeye_capability_schema": str(
                    robot_mask.handeye_capability_schema
                ),
                "handeye_calibration_tier": str(
                    robot_mask.handeye_calibration_tier
                ),
                "handeye_policy_sha256": str(
                    robot_mask.handeye_policy_sha256
                ),
                "handeye_calibration_status": str(
                    robot_mask.handeye_calibration_status
                ),
                "handeye_provisional": bool(robot_mask.handeye_provisional),
                "handeye_quality_gate_passed": (
                    robot_mask.handeye_quality_gate_passed
                ),
                "handeye_workspace_1sigma_max_m": (
                    robot_mask.handeye_workspace_1sigma_max_m
                ),
                "handeye_covariance_bound_transform": str(
                    robot_mask.handeye_covariance_bound_transform
                ),
                "handeye_covariance_scope": str(
                    robot_mask.handeye_covariance_scope
                ),
                "handeye_systematic_bias_status": str(
                    robot_mask.handeye_systematic_bias_status
                ),
                "handeye_systematic_bias_bound_m": (
                    robot_mask.handeye_systematic_bias_bound_m
                ),
                "handeye_autonomous_targeting_authorized": bool(
                    robot_mask.handeye_autonomous_targeting_authorized
                ),
                "handeye_provisional_policy_allowance_m": float(
                    robot_mask.handeye_provisional_policy_allowance_m
                ),
            },
            "robot_frame_index": copy.deepcopy(dict(persisted_index)),
        })
        return result

    def _submit_camera_capture(self, frame, *, source="direct"):
        if source not in {"direct", "delivery"}:
            raise ValueError("camera capture source must be direct or delivery")
        # Start switches direct polling to lossless FIFO while owning this same
        # lock. A direct call that observed the old flag cannot slip through
        # afterward and duplicate the FIFO's first frame.
        with self._ingress_submit_lock:
            if source == "direct" and self._camera_delivery_enabled:
                return True
            if (
                self._capture_ingress_paused
                or self.platform.state is not CollectionState.RECORDING
            ):
                return True
            boundary = self._recording_boundary
            if boundary is not None and (
                int(frame.index) <= int(boundary["camera_start_boundary_index"])
                or float(frame.t) < float(boundary["start_host_t_s"])
            ):
                return True
            evidence = None
            persisted_robot_index = None
            if self.require_robot:
                try:
                    evidence, persisted_robot_index = (
                        self._build_recorded_robot_evidence(frame)
                    )
                except RobotCaptureNotReady:
                    return None
                except Exception as exc:  # noqa: BLE001 - never write unbound RGB-D
                    self._robot_status = "ERROR"
                    self._robot_error = f"{type(exc).__name__}: {exc}"
                    self._abort_recording(
                        "left OpenArm frame evidence failed: "
                        f"{type(exc).__name__}: {exc}",
                        boundary_closed=True,
                    )
                    return False
            source_tracking = self._latest_tracking
            model_health = (
                self._safe_worker_health(self._model_worker, label="model")
                if self._model_worker is not None else None
            )
            if isinstance(model_health, dict):
                age = model_health.get("last_result_age_s")
                try:
                    age_current = age is not None and float(age) <= self.max_tracking_age_s
                except (TypeError, ValueError):
                    age_current = False
                if not (
                    model_health.get("status") == "OK"
                    and model_health.get("thread_alive") is not False
                    and age_current
                ):
                    source_tracking = {}
            if boundary is not None and isinstance(source_tracking, dict):
                try:
                    source_index = int(source_tracking.get("frame_index", 0))
                    source_t = float(source_tracking.get(
                        "camera_host_t_s", source_tracking.get("host_t"),
                    ))
                except (TypeError, ValueError):
                    source_index, source_t = 0, float("-inf")
                source_boundary = max(
                    int(boundary["camera_start_boundary_index"]),
                    int(boundary["model_start_boundary_index"]),
                )
                if (
                    source_index <= source_boundary
                    or source_t < float(boundary["start_host_t_s"])
                    or source_index > int(frame.index)
                    or source_t > float(frame.t)
                ):
                    source_tracking = {}
            try:
                tracking = associate_tracking_to_camera_frame(frame, source_tracking)
            except Exception as exc:
                self._abort_recording(
                    f"camera/tracking association failed: {type(exc).__name__}: {exc}",
                )
                return False
            capture_frame = frame
            if evidence is not None:
                capture_frame = evidence.camera_frame
                tracking = self._bind_robot_tracking_metadata(
                    tracking, evidence, persisted_robot_index,
                )
            accepted = self._submit_ingress(capture_frame, tracking)
            if accepted is True and evidence is not None:
                self._latest_recorded_robot_evidence = evidence
            return accepted

    def _consume_model_result(self):
        if self._model_worker is None:
            return False
        result = self._model_worker.latest()
        index = int(result.get("frame_index", 0))
        if index <= 0 and str(result.get("reason", "")) == "MODEL_STARTING":
            return False
        if index <= self._last_result_index:
            return False
        bound_frame = self._frame_ring.get(index)
        if bound_frame is None:
            # Latest-only inference may legitimately complete after its source
            # RGB-D frame ages out of the bounded association ring (notably on
            # first CUDA warm-up). Drop that result and fail the display closed;
            # resetting the camera or aborting an otherwise lossless recording
            # would convert harmless model latency into data loss.
            self._last_result_index = index
            self._latest_tracking = {
                "frame_index": index,
                "back_present": False,
                "initialized": False,
                "detector_confidence": 0.0,
                "reason": "MODEL_RESULT_EXPIRED",
                "points": [],
            }
            self._model_status = "CATCHING_UP"
            self._set_diagnostic_blockers([
                "perception result expired before camera-frame association",
            ])
            return True
        payload = tracking_payload(
            bound_frame, result,
            model_release_id=self.model_release_id,
            model_sha256=self.checkpoint_sha256,
        )
        payload["host_t"] = payload["camera_host_t_s"]
        payload["device_t"] = payload["camera_device_t_ms"]
        self._latest_tracking = payload
        self._pending_model_preview = (bound_frame, payload)
        self._last_result_index = index
        for old_index in list(self._frame_ring):
            if old_index <= index:
                self._frame_ring.pop(old_index, None)
        health = self._model_worker.health()
        if health["status"] == "ERROR":
            self._model_status = "ERROR"
            self._fatal_error = health["error"]
            self._abort_recording(f"perception model failed: {health['error']}")
        else:
            self._model_status = "RUNNING"
            self._fatal_error = None
            self._set_diagnostic_blockers([])
        return True

    def _select_preview_source(
        self, latest_frame, current_tracking, *, model_result_updated,
    ):
        """Select the exact RGB frame owned by a completed model result.

        A landmark result normally completes several camera frames late.  The
        preview must therefore display that retained source RGB frame, not draw
        the old points on a newer frame and not discard them merely because the
        live camera has advanced.
        """

        if self.fake or self._model_worker is None:
            return latest_frame, current_tracking
        if not model_result_updated:
            return None
        pending, self._pending_model_preview = self._pending_model_preview, None
        if pending is None:
            return latest_frame, current_tracking
        source_frame, source_tracking = pending
        if int(source_tracking.get("frame_index", -1)) != int(source_frame.index):
            return latest_frame, {
                "frame_index": int(latest_frame.index),
                "back_present": False,
                "initialized": False,
                "constellation_valid": False,
                "reason": "MODEL_SOURCE_FRAME_MISMATCH",
                "points": [],
            }
        source_depth_plausible = depth_back_plausible(
            source_frame.depth, source_frame.depth_scale,
        )
        source_effective = effective_tracking(
            source_tracking,
            expected_names=self.point_names,
            now=time.monotonic(),
            max_age_s=self.max_tracking_age_s,
            current_depth_plausible=source_depth_plausible,
        )
        return source_frame, source_effective

    def _loop(self):
        configuration_blockers = self._configuration_blockers()
        if configuration_blockers:
            self._fatal_error = "configuration blocked: " + "; ".join(configuration_blockers)
            self._camera_status = "CONFIGURATION_ERROR"
            self._model_status = "BLOCKED"
            self._set_diagnostic_blockers(configuration_blockers)
        else:
            self._attempt_camera_open()
        while not self._stop.is_set():
            # Advance the health epoch on any liveness degradation so an
            # in-flight Start can detect a worker that died mid-arm.
            self._observe_runtime_health()
            # Controller supervision (telemetry-fault -> disable) runs on the INDEPENDENT
            # watchdog thread, not here, so a hung/slow main loop never stalls it.
            if self.require_force:
                self._force_snapshot()
            self._consume_preview_result()
            if self._cam is None:
                if not configuration_blockers:
                    self._attempt_camera_open()
                frame_bgr = placeholder_frame(self._fatal_error or "camera unavailable")
                self._publish_preview(frame_bgr)
                remaining = max(0.0, self._next_camera_open_t - time.monotonic())
                time.sleep(min(0.1, remaining) if remaining > 0 else 0.01)
                continue
            model_fault = self._recover_failed_model_worker()
            if model_fault is not None:
                self._publish_preview(placeholder_frame(model_fault))
                continue
            # Lossless capture handoff is always serviced before model/overlay/
            # JPEG work. Preview and inference can drop obsolete work; capture
            # can only backpressure into the explicit bounded camera FIFO.
            self._drain_camera_delivery()
            try:
                frame = self._cam.latest_frame()
            except Exception as exc:
                reason = f"camera read failed: {type(exc).__name__}: {exc}"
                self._reset_camera_pipeline(reason)
                self._publish_preview(placeholder_frame(reason))
                continue
            if frame.status != "OK" or frame.frame is None:
                self._camera_status = str(frame.status)
                if str(frame.status) in {"STALE", "ERROR"}:
                    reason = f"camera became {frame.status}; reconnecting"
                    self._reset_camera_pipeline(reason)
                    self._publish_preview(placeholder_frame(reason), frame=frame)
                    continue
                frame_bgr = placeholder_frame(
                    f"camera {frame.status.lower()} — recording blocked"
                )
                self._publish_preview(frame_bgr, frame=frame)
                time.sleep(0.03)
                continue

            self._camera_status = "OK"
            new_camera_frame = frame.index != self._last_camera_index
            if new_camera_frame:
                self._last_camera_index = int(frame.index)
                detached = clone_camera_frame(frame)
                self._current_depth_plausible = self.fake or depth_back_plausible(
                    detached.depth, detached.depth_scale,
                )
                if self.fake:
                    if self._model_status != "RUNNING":
                        self._update_provenance()
                        self._model_status = "RUNNING"
                        self._set_diagnostic_blockers([])
                    tracking = self._fake_tracking(detached)
                    self._latest_tracking = tracking
                    self._last_result_index = int(frame.index)
                    if not self._camera_delivery_enabled:
                        self._submit_camera_capture(detached)
                else:
                    # Camera-rate recording is independent from the latest-only
                    # model worker. Associate the newest completed model result,
                    # then enqueue every acquired RGB-D frame exactly once.
                    if not self._camera_delivery_enabled:
                        self._submit_camera_capture(detached)
                    # Latest-only inference may discard an older pending frame,
                    # but the selected frame itself cannot run until its exact
                    # same-frame OpenArm mask bundle exists.
                    self._pending_model_frame = detached
            model_input_submitted = False
            if not self.fake and self._pending_model_frame is not None:
                pending_model_frame = self._pending_model_frame
                try:
                    model_input = self._model_input_for_frame(
                        pending_model_frame
                    )
                except RobotCaptureNotReady:
                    self._model_status = "WAITING_ROBOT"
                except Exception as exc:
                    self._reset_camera_pipeline(
                        "robot/model frame evidence failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                    continue
                else:
                    if self._model_worker is None and self._fatal_error is None:
                        try:
                            processor = self._build_tracker_processor(
                                pending_model_frame
                            )
                            # Validate the live device identity and bind negotiated mode
                            # before any worker can publish a healthy model result.
                            self._update_provenance()
                            self._model_worker = LatestFrameWorker(
                                processor, name="wam-v9-tracker",
                            ).start()
                            self._model_status = "WARMING_UP"
                        except Exception as exc:
                            self._reset_camera_pipeline(
                                f"model init failed: {type(exc).__name__}: {exc}",
                            )
                            continue
                    if self._model_worker is not None:
                        bound_frame = getattr(
                            model_input, "camera_frame", pending_model_frame,
                        )
                        self._frame_ring[int(bound_frame.index)] = bound_frame
                        while len(self._frame_ring) > 32:
                            self._frame_ring.popitem(last=False)
                        self._model_worker.submit(model_input)
                        self._pending_model_frame = None
                        model_input_submitted = True
            model_result_updated = self._consume_model_result()
            model_fault = self._recover_failed_model_worker()
            if model_fault is not None:
                self._publish_preview(placeholder_frame(model_fault))
                continue
            if (
                not new_camera_frame
                and not model_input_submitted
                and not model_result_updated
            ):
                time.sleep(0.002)
                continue

            current_tracking = effective_tracking(
                self._latest_tracking,
                expected_names=self.point_names,
                now=time.monotonic(),
                max_age_s=self.max_tracking_age_s,
                current_depth_plausible=self._current_depth_plausible,
            )
            with self._lock:
                self._current_tracking = current_tracking
            self.platform.update_tracking(current_tracking, frame=frame)
            if not self.fake and self._model_worker is not None:
                model_health = self._model_worker.health()
                result_age = model_health.get("last_result_age_s")
                if model_health.get("status") == "ERROR":
                    model_fault = self._recover_failed_model_worker()
                    if model_fault is not None:
                        self._publish_preview(placeholder_frame(model_fault))
                        continue
                elif result_age is None or result_age > self.max_tracking_age_s:
                    self._model_status = "STALE" if result_age is not None else "WARMING_UP"
                    self._set_diagnostic_blockers(["perception model output is not current"])
                    # Lossless RGB-D capture is independent from latest-only
                    # inference. Staleness hides points/Confirm and is recorded
                    # as an unavailable model association; it must not destroy a
                    # long acquisition.
            ingress_health = self._ingress_worker.health()
            if ingress_health.get("status") == "ERROR":
                self._set_diagnostic_blockers([
                    f"diagnostic capture ingress failed: {ingress_health.get('error')}",
                ])
                self._abort_recording(
                    f"diagnostic capture ingress failed: {ingress_health.get('error')}",
                )
            elif (
                self.platform.state is CollectionState.RECORDING
                and int(ingress_health.get("dropped_before_inference", 0))
                > self._record_ingress_drop_baseline
            ):
                dropped = (
                    int(ingress_health["dropped_before_inference"])
                    - self._record_ingress_drop_baseline
                )
                self._abort_recording(
                    f"diagnostic capture ingress dropped {dropped} model-bound frame(s)"
                )
            snapshot = self.platform.state_snapshot()
            completed_names = {
                self.point_names[index]
                for index in snapshot["completed"]
                if 0 <= index < len(self.point_names)
            }
            preview_source = self._select_preview_source(
                frame,
                current_tracking,
                model_result_updated=model_result_updated,
            )
            if preview_source is not None:
                preview_frame, preview_tracking = preview_source
                self._submit_preview(
                    preview_frame,
                    preview_tracking,
                    active_name=snapshot["point"],
                    completed_names=completed_names,
                )
            self._consume_preview_result()
            time.sleep(0.005)

    def _render_preview_bundle(self, bundle):
        """Render a disposable latest-only doctor preview at bounded resolution."""

        import cv2

        frame_bgr = np.ascontiguousarray(bundle.frame[:, :, ::-1])
        draw_tracking_overlay(
            frame_bgr, bundle.tracking,
            active_name=bundle.active_name,
            completed_names=bundle.completed_names,
            expected_names=self.point_names,
        )
        height, width = frame_bgr.shape[:2]
        if width > 960:
            preview_width = 960
            preview_height = max(1, int(round(height * preview_width / width)))
            frame_bgr = cv2.resize(
                frame_bgr, (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )
        return {
            "frame_index": int(bundle.index),
            "preview_host_t_s": float(bundle.t),
            "jpeg": encode_jpeg(frame_bgr),
        }

    def _submit_preview(self, frame, tracking, *, active_name, completed_names):
        if self._preview_worker is None:
            return False
        bundle = SimpleNamespace(
            index=int(frame.index),
            t=float(frame.t),
            frame=frame.frame,
            tracking=tracking_for_preview_frame(frame.index, tracking),
            active_name=str(active_name),
            completed_names=tuple(completed_names),
        )
        return self._preview_worker.submit(bundle)

    def _consume_preview_result(self):
        if self._preview_worker is None:
            return False
        result = self._preview_worker.latest()
        try:
            index = int(result.get("frame_index", -1))
        except (TypeError, ValueError):
            return False
        encoded = result.get("jpeg")
        if index <= self._preview_result_index or not isinstance(encoded, bytes):
            return False
        preview_t = float(result.get("preview_host_t_s"))
        published_t = time.monotonic()
        with self._lock:
            self.jpeg = encoded
            self._preview_frame_index = index
            self._preview_frame_t = preview_t
            self._preview_frame_confirmable = True
            self._preview_result_index = index
            self._preview_publish_times.append(published_t)
        return True

    def _publish_preview(self, frame_bgr, frame=None):
        encoded = encode_jpeg(frame_bgr)
        with self._lock:
            self.jpeg = encoded
            self._preview_frame_confirmable = False
            if frame is not None:
                self._preview_frame_index = int(frame.index)
                self._preview_frame_t = float(frame.t)

    def _validate_completion_result(self, result, session_dir):
        """Make COMPLETE itself depend on compact on-disk durability proof."""

        if Path(str(result.get("session_dir", ""))).resolve() != Path(session_dir).resolve():
            raise RuntimeError("diagnostic finalize result session path mismatch")
        try:
            n_rgb = int(result["n_rgb_frames"])
            decoded = int(result["decoded_rgb_frames"])
            n_depth = int(result["n_depth_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("diagnostic finalize result counts are incomplete") from exc
        if n_rgb <= 0 or decoded != n_rgb or n_depth != n_rgb:
            raise RuntimeError("diagnostic RGB/decode/depth counts are not exact")
        summary = self._verified_finalization_summary(
            {"session_dir": str(Path(session_dir).resolve())}, result,
        )
        if summary.get("complete") is not True:
            raise RuntimeError("diagnostic manifest/video/seal verification failed")
        timebase = result.get("video_timebase")
        if not isinstance(timebase, dict) or timebase.get("verified") is not True:
            raise RuntimeError("diagnostic video timebase is not verified")
        depth = result.get("depth_integrity")
        if not isinstance(depth, dict) or int(depth.get("count", -1)) != n_depth:
            raise RuntimeError("diagnostic depth integrity count is incomplete")
        if depth.get("path") != "depth_integrity.jsonl":
            raise RuntimeError("diagnostic depth integrity path is invalid")
        depth_path = Path(session_dir) / "depth_integrity.jsonl"
        depth_sha, depth_bytes, _identity = _verified_regular_file(depth_path)
        if (
            depth_sha != depth.get("index_sha256")
            or depth_bytes <= 0
            or depth.get("dtype") != "uint16"
        ):
            raise RuntimeError("diagnostic depth integrity index verification failed")
        if self.audio_device is not None:
            audio = result.get("audio")
            if (
                not isinstance(audio, dict)
                or audio.get("path") != "audio.wav"
                or audio.get("status") != "OK"
                or int(audio.get("sample_rate", -1)) != 16_000
                or int(audio.get("input_sample_rate", -1)) != self.audio_sample_rate
                or int(audio.get("n_samples", 0)) <= 0
            ):
                raise RuntimeError("diagnostic audio summary is incomplete")
            audio_path = Path(session_dir) / "audio.wav"
            audio_sha, audio_bytes, _audio_identity = _verified_regular_file(audio_path)
            if audio_bytes <= 0 or audio_sha != audio.get("sha256"):
                raise RuntimeError("diagnostic audio WAV verification failed")
        if self.require_force:
            force = result.get("force")
            if not isinstance(force, dict):
                raise RuntimeError("diagnostic Hall-force summary is incomplete")
            from mechanical_sensor.force_feedback.force_capture import (
                verify_force_capture,
            )
            verified_force = verify_force_capture(session_dir)
            expected_force = {
                "path": "force.csv",
                "t_start": float(verified_force.t_start),
                "t_end": float(verified_force.t_end),
                "n_samples": int(verified_force.n_samples),
                "status": str(verified_force.status),
                "sha256": str(verified_force.sha256),
                "device_t_start_ms": float(verified_force.device_t_start_ms),
                "device_t_end_ms": float(verified_force.device_t_end_ms),
                "calibration_sha256": str(verified_force.calibration_sha256),
                "device_identity_sha256": str(
                    verified_force.device_identity_sha256
                ),
            }
            if force != expected_force:
                raise RuntimeError(
                    "diagnostic Hall-force summary does not match sealed artifacts"
                )
        stitch = result.get("video_stitch")
        provenance = stitch.get("provenance") if isinstance(stitch, dict) else None
        packet_validation = (
            stitch.get("packet_validation") if isinstance(stitch, dict) else None
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("method") != "ffmpeg_concat_demuxer_stream_copy"
            or provenance.get("sha256") != self.ffmpeg_sha256
            or Path(str(provenance.get("resolved_path", ""))).resolve()
            != Path(str(self.ffmpeg_path)).expanduser().resolve()
            or provenance.get("shell") is not False
        ):
            raise RuntimeError("diagnostic concat-copy binary provenance is invalid")
        if (
            not isinstance(packet_validation, dict)
            or packet_validation.get("verified") is not True
            or packet_validation.get("method") != "ffmpeg_decode_null_muxer"
            or int(packet_validation.get("decoded_frame_count", -1)) != n_rgb
            or int(packet_validation.get("expected_frame_count", -1)) != n_rgb
        ):
            raise RuntimeError("diagnostic final video decode integrity is invalid")
        boundary = result.get("capture_boundary")
        if not isinstance(boundary, dict):
            raise RuntimeError("diagnostic capture boundary proof is missing")
        start = boundary.get("camera_delivery_start_proof")
        end = boundary.get("camera_end_boundary_proof")
        camera_fifo = boundary.get("camera_delivery_fifo")
        ingress = boundary.get("capture_ingress_fifo")
        if (
            not isinstance(start, dict)
            or start.get("boundary_latched_atomically") is not True
            or not isinstance(end, dict)
            or end.get("boundary_latched_atomically") is not True
            or end.get("post_boundary_overwrite_prevented") is not True
        ):
            raise RuntimeError("diagnostic Start/End atomic proof is invalid")
        if not isinstance(camera_fifo, dict) or (
            camera_fifo.get("status") != "OK"
            or camera_fifo.get("interval_active") is not False
            or camera_fifo.get("accepting") is not False
            or int(camera_fifo.get("queue_depth", -1)) != 0
            or int(camera_fifo.get("interval_overflow_frames", -1)) != 0
            or int(camera_fifo.get("interval_enqueued_frames", -1))
            != int(camera_fifo.get("interval_delivered_frames", -2))
        ):
            raise RuntimeError("diagnostic camera delivery FIFO proof is invalid")
        if not isinstance(ingress, dict) or (
            ingress.get("status") != "OK"
            or ingress.get("interval_active") is not False
            or int(ingress.get("queue_depth", -1)) != 0
            or int(ingress.get("overflow_rejections", -1)) != 0
            or int(ingress.get("abandoned_after_error", -1)) != 0
            or int(ingress.get("accepted", -1)) != int(ingress.get("processed", -2))
            or int(ingress.get("last_accepted_index", -1))
            != int(ingress.get("last_completed_index", -2))
        ):
            raise RuntimeError("diagnostic disk ingress FIFO proof is invalid")
        closure = validate_capture_boundary_closure(
            boundary,
            n_rgb_frames=n_rgb,
            expected_fps=self._negotiated_camera_fps(),
            final_frame_range=result.get("frame_index_range"),
        )
        if boundary.get("index_closure") != closure:
            raise RuntimeError("diagnostic capture boundary closure proof changed")

    def _verified_finalization_summary(self, snapshot, finalization):
        seal_memory = finalization.get("diagnostic_finalization")
        seal_compact = finalization.get("diagnostic_seal")
        session_path = Path(snapshot["session_dir"]).resolve()
        video_path = session_path / "rgb.mp4"
        manifest_path = session_path / "manifest.json"
        seal_path = session_path / "DIAGNOSTIC_FINALIZATION.json"
        try:
            n_rgb = int(finalization.get("n_rgb_frames", 0))
        except (TypeError, ValueError):
            n_rgb = 0
        decoded_raw = finalization.get("decoded_rgb_frames")
        try:
            decoded_rgb = None if decoded_raw is None else int(decoded_raw)
        except (TypeError, ValueError):
            decoded_rgb = None
        summary = {
            "complete": False,
            "rgb_mp4_path": str(video_path),
            "n_rgb_frames": n_rgb,
            "decoded_rgb_frames": decoded_rgb,
            "n_depth_frames": finalization.get("n_depth_frames"),
            "rgb_mp4_sha256": None,
            "diagnostic_seal_path": str(seal_path),
            "capture_boundary": finalization.get("capture_boundary"),
            "depth_integrity": finalization.get("depth_integrity"),
            "segment_validation": finalization.get("segment_validation"),
            "timeline": finalization.get("timeline"),
            "video_stitch": finalization.get("video_stitch"),
            "video_timebase": finalization.get("video_timebase"),
            "storage": finalization.get("storage"),
            "audio": finalization.get("audio"),
            "robot": finalization.get("robot"),
        }
        if finalization.get("audio") is not None or finalization.get(
            "audio_authenticity"
        ) is not None:
            summary["audio_authenticity"] = finalization.get(
                "audio_authenticity"
            )
        if self.require_force or finalization.get("force") is not None:
            summary["force"] = finalization.get("force")
            summary["force_authenticity"] = finalization.get(
                "force_authenticity"
            )
        if (
            finalization.get("quality") is not None
            or finalization.get("quality_authenticity") is not None
        ):
            summary["quality"] = finalization.get("quality")
            summary["quality_authenticity"] = finalization.get(
                "quality_authenticity"
            )
        try:
            seal_raw = seal_path.read_bytes()
            seal_disk = json.loads(seal_raw.decode("utf-8"))
            if not isinstance(seal_disk, dict):
                return summary
            if isinstance(seal_memory, dict) and seal_disk != seal_memory:
                return summary
            if isinstance(seal_compact, dict) and (
                seal_compact.get("path") != "DIAGNOSTIC_FINALIZATION.json"
                or seal_compact.get("manifest_sha256") != seal_disk.get("manifest_sha256")
                or int(seal_compact.get("artifact_count", -1))
                != len(seal_disk.get("artifacts", {}))
            ):
                return summary
            artifacts = seal_disk.get("artifacts", {})
            video_artifact = artifacts.get("rgb.mp4", {}) if isinstance(artifacts, dict) else {}
            if not isinstance(video_artifact, dict):
                return summary
            video_sha = video_artifact.get("sha256")
            summary["rgb_mp4_sha256"] = video_sha
            if (
                n_rgb <= 0
                or decoded_rgb != n_rgb
                or not isinstance(video_sha, str)
                or len(video_sha) != 64
                or any(character not in "0123456789abcdef" for character in video_sha)
            ):
                return summary
            # DiagnosticPerceptionCapture's canonical artifact inventory names
            # the on-disk byte count ``bytes``. Do not accept an alternate key:
            # the UI proof must verify the exact seal contract it displays.
            expected_size = int(video_artifact.get("bytes", -1))
            if expected_size <= 0:
                return summary
            manifest_sha = seal_disk.get("manifest_sha256")
            if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
                return summary
            actual_manifest_sha, _manifest_size, _manifest_identity = (
                _verified_regular_file(manifest_path)
            )
            if actual_manifest_sha != manifest_sha:
                return summary
            video_identity = _regular_file_identity(video_path)
            if int(video_identity[2]) != expected_size:
                return summary
            audio_summary = finalization.get("audio")
            audio_cache_rows = []
            if self.audio_device is not None and not isinstance(audio_summary, dict):
                return summary
            if isinstance(audio_summary, dict):
                audio_authenticity = finalization.get("audio_authenticity")
                if (
                    seal_disk.get("audio") != audio_summary
                    or seal_disk.get("audio_authenticity") != audio_authenticity
                    or not isinstance(audio_authenticity, dict)
                    or audio_authenticity.get("schema_version")
                    != "wam_audio_signal_authenticity_v1"
                    or audio_authenticity.get("policy_version")
                    != "audio_signal_conservative_v2"
                    or audio_authenticity.get("status") != "PASS"
                ):
                    return summary
                try:
                    audio_metadata = json.loads(
                        (session_path / "audio_capture.json").read_text(
                            encoding="utf-8"
                        )
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    return summary
                if (
                    not isinstance(audio_metadata, dict)
                    or audio_metadata.get("signal_authenticity")
                    != audio_authenticity
                ):
                    return summary
                if (
                    audio_summary.get("path") != "audio.wav"
                    or audio_summary.get("status") != "OK"
                    or int(audio_summary.get("sample_rate", -1)) != 16_000
                    or int(audio_summary.get("input_sample_rate", -1))
                    != self.audio_sample_rate
                    or int(audio_summary.get("n_samples", 0)) <= 0
                ):
                    return summary
                for name in (
                    "audio.wav", "audio_capture.json",
                    "audio_input_times.jsonl", "audio_times.jsonl",
                ):
                    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
                    if not isinstance(artifact, dict):
                        return summary
                    expected_audio_sha = artifact.get("sha256")
                    expected_audio_size = int(artifact.get("bytes", -1))
                    if (
                        not isinstance(expected_audio_sha, str)
                        or len(expected_audio_sha) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in expected_audio_sha
                        )
                        or expected_audio_size <= 0
                    ):
                        return summary
                    if name == "audio.wav" and (
                        audio_summary.get("sha256") != expected_audio_sha
                    ):
                        return summary
                    identity = _regular_file_identity(session_path / name)
                    if int(identity[2]) != expected_audio_size:
                        return summary
                    audio_cache_rows.append(
                        (name, expected_audio_sha, expected_audio_size, identity)
                    )
            robot_summary = finalization.get("robot")
            robot_cache_rows = []
            if self.require_robot and not isinstance(robot_summary, dict):
                return summary
            if isinstance(robot_summary, dict):
                try:
                    robot_contract = self._actual_robot_activation_contract(
                        self._robot_coordinator
                    )
                except RuntimeError:
                    return summary
                if (
                    seal_disk.get("robot") != robot_summary
                    or not _robot_v4_summary_valid(
                        robot_summary, contract=robot_contract,
                    )
                    or robot_summary.get("side") != "left"
                    or robot_summary.get("controller_mode") != "unilateral"
                    or robot_summary.get("leader_can_interface") != "can1"
                    or robot_summary.get("follower_can_interface") != "can3"
                    or int(robot_summary.get("frame_count", -1)) != n_rgb
                ):
                    return summary
                inventory = robot_summary.get("hash_inventory")
                if not isinstance(inventory, dict):
                    return summary
                robot_names = (
                    "robot_control_envelopes.jsonl",
                    "robot_exact_control.csv",
                    "robot_frame_index.jsonl",
                    "robot_capture_summary.json",
                )
                for name in robot_names:
                    artifact = (
                        artifacts.get(name)
                        if isinstance(artifacts, dict) else None
                    )
                    if not isinstance(artifact, dict):
                        return summary
                    expected_robot_sha = artifact.get("sha256")
                    expected_robot_size = int(artifact.get("bytes", -1))
                    if (
                        not isinstance(expected_robot_sha, str)
                        or re.fullmatch(r"[0-9a-f]{64}", expected_robot_sha)
                        is None
                        or expected_robot_size <= 0
                    ):
                        return summary
                    if name != "robot_capture_summary.json":
                        item = inventory.get(name)
                        if (
                            not isinstance(item, dict)
                            or item.get("sha256") != expected_robot_sha
                            or int(item.get("size_bytes", -1))
                            != expected_robot_size
                        ):
                            return summary
                    identity = _regular_file_identity(session_path / name)
                    if int(identity[2]) != expected_robot_size:
                        return summary
                    robot_cache_rows.append(
                        (
                            name,
                            expected_robot_sha,
                            expected_robot_size,
                            identity,
                        )
                    )
                try:
                    disk_robot_summary = json.loads(
                        (
                            session_path / "robot_capture_summary.json"
                        ).read_text(encoding="utf-8")
                    )
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    return summary
                if disk_robot_summary != robot_summary:
                    return summary
            force_summary = finalization.get("force")
            force_authenticity = finalization.get("force_authenticity")
            force_cache_rows = []
            if self.require_force and force_summary is None:
                return summary
            if (force_summary is None) != (force_authenticity is None):
                return summary
            if force_summary is not None:
                if (
                    not isinstance(force_summary, dict)
                    or not isinstance(force_authenticity, dict)
                ):
                    return summary
                if (
                    seal_disk.get("force") != force_summary
                    or seal_disk.get("force_authenticity") != force_authenticity
                    or force_authenticity.get("schema_version")
                    != "wam_force_signal_authenticity_v1"
                    or force_authenticity.get("policy_version")
                    != "force_signal_conservative_v3"
                    or force_authenticity.get("status") != "PASS"
                ):
                    return summary
                from mechanical_sensor.force_feedback.force_capture import (
                    verify_force_capture,
                )
                verified_force = verify_force_capture(session_path)
                if verified_force.signal_authenticity != force_authenticity:
                    return summary
                force_checks = force_authenticity.get("checks")
                if not isinstance(force_checks, dict):
                    return summary
                calibration_recomputation = force_checks.get(
                    "calibration_recomputation"
                )
                if calibration_recomputation == "PASS":
                    if (
                        force_authenticity.get("promotion_eligible") is not True
                        or force_authenticity.get("promotion_blockers") != []
                    ):
                        return summary
                    force_artifact_names = (
                        "force.csv",
                        "force_capture.json",
                        "FORCE_CAPTURE_COMPLETE.json",
                        "force_calibration.json",
                    )
                elif calibration_recomputation == "NOT_AVAILABLE":
                    if (
                        force_authenticity.get("promotion_eligible") is not False
                        or "force_calibration.json" in artifacts
                    ):
                        return summary
                    force_artifact_names = (
                        "force.csv",
                        "force_capture.json",
                        "FORCE_CAPTURE_COMPLETE.json",
                    )
                else:
                    return summary
                if force_summary != {
                    "path": "force.csv",
                    "t_start": float(verified_force.t_start),
                    "t_end": float(verified_force.t_end),
                    "n_samples": int(verified_force.n_samples),
                    "status": str(verified_force.status),
                    "sha256": str(verified_force.sha256),
                    "device_t_start_ms": float(verified_force.device_t_start_ms),
                    "device_t_end_ms": float(verified_force.device_t_end_ms),
                    "calibration_sha256": str(verified_force.calibration_sha256),
                    "device_identity_sha256": str(
                        verified_force.device_identity_sha256
                    ),
                }:
                    return summary
                for name in force_artifact_names:
                    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
                    if not isinstance(artifact, dict):
                        return summary
                    expected_force_sha = artifact.get("sha256")
                    expected_force_size = int(artifact.get("bytes", -1))
                    if (
                        not isinstance(expected_force_sha, str)
                        or re.fullmatch(r"[0-9a-f]{64}", expected_force_sha) is None
                        or expected_force_size <= 0
                    ):
                        return summary
                    identity = _regular_file_identity(session_path / name)
                    if int(identity[2]) != expected_force_size:
                        return summary
                    force_cache_rows.append(
                        (name, expected_force_sha, expected_force_size, identity)
                    )
            quality_summary = finalization.get("quality")
            quality_authenticity = finalization.get("quality_authenticity")
            quality_cache_rows = []
            if (quality_summary is None) != (quality_authenticity is None):
                return summary
            if quality_summary is not None:
                if (
                    not isinstance(quality_summary, dict)
                    or not isinstance(quality_authenticity, dict)
                    or force_summary is None
                ):
                    return summary
                if (
                    seal_disk.get("quality") != quality_summary
                    or seal_disk.get("quality_authenticity")
                    != quality_authenticity
                ):
                    return summary
                from ausc.realtime_q import (
                    quality_packet_log_summary,
                    verify_quality_packet_log,
                )
                verified_quality = verify_quality_packet_log(session_path)
                if (
                    verified_quality != quality_authenticity
                    or quality_packet_log_summary(verified_quality)
                    != quality_summary
                ):
                    return summary
                raw_quality_inputs = verified_quality.get(
                    "model_input_inventory", {}
                ).get("files")
                if not isinstance(raw_quality_inputs, list) or not raw_quality_inputs:
                    return summary
                quality_names = [
                    "quality_inference_packets.jsonl",
                    "quality_inference_authenticity.json",
                ]
                for claim in raw_quality_inputs:
                    if not isinstance(claim, dict):
                        return summary
                    path = claim.get("path")
                    if (
                        not isinstance(path, str)
                        or len(Path(path).parts) != 2
                        or Path(path).parts[0] != "quality_model_inputs"
                    ):
                        return summary
                    quality_names.append(path)
                for name in quality_names:
                    artifact = (
                        artifacts.get(name)
                        if isinstance(artifacts, dict) else None
                    )
                    if not isinstance(artifact, dict):
                        return summary
                    expected_quality_sha = artifact.get("sha256")
                    expected_quality_size = int(artifact.get("bytes", -1))
                    if (
                        not isinstance(expected_quality_sha, str)
                        or re.fullmatch(r"[0-9a-f]{64}", expected_quality_sha)
                        is None
                        or expected_quality_size <= 0
                    ):
                        return summary
                    identity = _regular_file_identity(session_path / name)
                    if int(identity[2]) != expected_quality_size:
                        return summary
                    quality_cache_rows.append((
                        name,
                        expected_quality_sha,
                        expected_quality_size,
                        identity,
                    ))
            cache_key = (
                str(session_path), video_sha, expected_size, video_identity,
                tuple(audio_cache_rows),
                tuple(robot_cache_rows),
                tuple(force_cache_rows),
                tuple(quality_cache_rows),
            )
            if self._completion_proof_cache != cache_key:
                actual_video_sha, actual_size, verified_identity = (
                    _verified_regular_file(video_path)
                )
                if (
                    actual_video_sha != video_sha
                    or actual_size != expected_size
                    or verified_identity != video_identity
                ):
                    return summary
                for name, expected_audio_sha, expected_audio_size, identity in audio_cache_rows:
                    actual_sha, actual_size, verified_audio_identity = (
                        _verified_regular_file(session_path / name)
                    )
                    if (
                        actual_sha != expected_audio_sha
                        or actual_size != expected_audio_size
                        or verified_audio_identity != identity
                    ):
                        return summary
                for (
                    name,
                    expected_robot_sha,
                    expected_robot_size,
                    identity,
                ) in robot_cache_rows:
                    actual_sha, actual_size, verified_robot_identity = (
                        _verified_regular_file(session_path / name)
                    )
                    if (
                        actual_sha != expected_robot_sha
                        or actual_size != expected_robot_size
                        or verified_robot_identity != identity
                    ):
                        return summary
                for name, expected_force_sha, expected_force_size, identity in force_cache_rows:
                    actual_sha, actual_size, verified_force_identity = (
                        _verified_regular_file(session_path / name)
                    )
                    if (
                        actual_sha != expected_force_sha
                        or actual_size != expected_force_size
                        or verified_force_identity != identity
                    ):
                        return summary
                for (
                    name,
                    expected_quality_sha,
                    expected_quality_size,
                    identity,
                ) in quality_cache_rows:
                    actual_sha, actual_size, verified_quality_identity = (
                        _verified_regular_file(session_path / name)
                    )
                    if (
                        actual_sha != expected_quality_sha
                        or actual_size != expected_quality_size
                        or verified_quality_identity != identity
                    ):
                        return summary
                self._completion_proof_cache = cache_key
            summary["complete"] = True
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError,
                TypeError, ValueError):
            return summary
        return summary

    def _robot_snapshot(self):
        base = {
            "enabled": self.require_robot,
            "required": self.require_robot,
            "pair": "left:can1->can3" if self.require_robot else None,
            "status": self._robot_status,
            "ready": not self.require_robot,
            "recording": False,
            "camera_arm_authorized": False,
            "pending_camera_frame": (
                None
                if self._pending_delivery_frame is None
                else int(self._pending_delivery_frame.index)
            ),
            "error": self._robot_error,
        }
        coordinator = self._robot_coordinator
        if coordinator is None:
            if self.require_robot:
                base.update({
                    "status": "MISSING",
                    "ready": False,
                    "error": "left OpenArm capture coordinator is unavailable",
                })
            return base
        base["recording"] = bool(getattr(coordinator, "recording", False))
        base["camera_arm_authorized"] = bool(
            getattr(coordinator, "camera_arm_authorized", False)
        )
        if bool(getattr(coordinator, "faulted", False)):
            base.update({
                "status": "ERROR",
                "ready": False,
                "error": str(
                    getattr(coordinator, "fault_reason", "")
                    or "left OpenArm coordinator fault"
                ),
            })
            return base
        try:
            self._actual_robot_activation_contract(coordinator)
        except RuntimeError as exc:
            base.update({
                "status": "ERROR",
                "ready": False,
                "error": str(exc),
            })
            return base
        if (
            getattr(coordinator, "telemetry_schema", None)
            != _ROBOT_ENVELOPE_SCHEMA_V4
            or getattr(coordinator, "telemetry_schemas", None)
            != frozenset({_ROBOT_ENVELOPE_SCHEMA_V4})
        ):
            base.update({
                "status": "ERROR",
                "ready": False,
                "error": "left OpenArm production telemetry must be v4-only",
            })
            return base
        health_getter = getattr(coordinator, "receiver_health", None)
        if not callable(health_getter):
            base.update({
                "status": "WAITING",
                "ready": False,
                "error": "left OpenArm receiver health is unavailable",
            })
            return base
        try:
            health = health_getter()
            if is_dataclass(health):
                payload = asdict(health)
            elif isinstance(health, Mapping):
                payload = dict(health)
            elif hasattr(health, "__dict__"):
                payload = dict(vars(health))
            else:
                raise RuntimeError("receiver health is not serializable")
            ready = (
                payload.get("preflight_ready") is True
                and payload.get("pair") == "left:can1->can3"
                and str(payload.get("status")) in {"READY", "OK"}
            )
            base.update({
                "status": "READY" if ready else str(
                    payload.get("status") or "WAITING"
                ),
                "ready": ready,
                "receiver": payload,
                "error": None if ready else payload.get("error"),
            })
        except Exception as exc:  # health can never be assumed
            base.update({
                "status": "ERROR",
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        return base

    def _handeye_snapshot(self):
        handeye = self._handeye_activation
        if handeye is None:
            return {"status": "MISSING"}
        provisional = bool(getattr(handeye, "provisional", False))
        snapshot = {
            "status": "BOUND_PROVISIONAL" if provisional else "BOUND_VALIDATED",
            "release_id": str(getattr(handeye, "handeye_release_id")),
            "artifact_sha256": str(getattr(handeye, "handeye_sha256")),
            "calibration_status": str(
                getattr(
                    handeye,
                    "calibration_status",
                    getattr(handeye, "status", "UNKNOWN"),
                )
            ),
            "provisional": provisional,
            "quality_gate_passed": bool(
                getattr(handeye, "quality_gate_passed", not provisional)
            ),
            "workspace_1sigma_median_mm": (
                float(handeye.workspace_1sigma_median) * 1000.0
                if hasattr(handeye, "workspace_1sigma_median") else None
            ),
            "workspace_1sigma_max_mm": (
                float(handeye.workspace_1sigma_max) * 1000.0
                if hasattr(handeye, "workspace_1sigma_max") else None
            ),
            "authorized_scopes": list(
                getattr(handeye, "authorized_scopes", ())
            ),
            "autonomous_targeting_authorized": bool(
                getattr(
                    handeye,
                    "autonomous_targeting_authorized",
                    not provisional,
                )
            ),
            "source_frame": getattr(handeye, "source_frame", None),
            "target_frame": getattr(handeye, "target_frame", None),
        }
        return snapshot

    def _audio_snapshot(self):
        base = {
            "enabled": self.audio_device is not None,
            "required": self.require_audio,
            "device": self.audio_device,
            "input_sample_rate": self.audio_sample_rate,
            "model_sample_rate": 16_000,
            "channels": self.audio_channels,
            "pick_channel": self.audio_pick_channel,
            "hardware": copy.deepcopy(self._audio_hardware),
            "status": self._audio_status,
            "rms_db": -120.0,
            "peak_db": -120.0,
            "recording": self._audio_recording_active,
            "error": self._audio_error,
            "commissioned": False,
            "signal_status": "DISABLED" if self.audio_device is None else "WAITING",
            "signal_present": False,
        }
        if self._audio is None:
            return base
        try:
            level = self._audio.level_db()
            base.update({
                "status": str(level.status),
                "rms_db": round(float(level.rms_db), 1),
                "peak_db": round(float(level.peak_db), 1),
                "error": None,
                "signal_status": (
                    "PRESENT"
                    if str(level.status) == "OK" and float(level.peak_db) >= AUDIO_FLOOR_DB
                    else "SILENT"
                    if str(level.status) == "OK"
                    else str(level.status)
                ),
                "signal_present": (
                    str(level.status) == "OK" and float(level.peak_db) >= AUDIO_FLOOR_DB
                ),
            })
        except Exception as exc:  # noqa: BLE001 - audio health is user-visible
            base.update({
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })
        return base

    def _on_force_safety_event(self, sample):
        # P0-4: the Hall force reading is an advisory reference for physicians, NOT a
        # safety interlock. An over-threshold reading updates the displayed status only
        # -- it never aborts recording and never touches the arm. Contact-force safety
        # is mechanical (passive-compliance mount) + human teleop + the controller
        # E-stop/disable path, none of which depend on this reading.
        status = str(getattr(sample, "status", "UNSAFE"))
        force_n = float(getattr(sample, "force_n", 0.0))
        self._force_status = status
        self._force_error = f"Hall-force {status} at {force_n:.3f} N (advisory)"

    def _on_force_capture_fault(self, error, observed_t=None):
        # P0-4: a Hall sensor dropout degrades the advisory force channel but never
        # aborts the session or touches the arm; every other modality keeps recording
        # and the force channel is honestly marked degraded.
        del observed_t
        self._force_status = "ERROR"
        self._force_error = (
            "Hall-force serial failed (advisory channel degraded): "
            f"{type(error).__name__}: {error}"
        )

    def _force_snapshot(self):
        base = {
            "enabled": self._force is not None,
            "required": self.require_force,
            "monitor_required": self.commissioning_hall_monitor_only,
            "mode": (
                "UNSYNCED_COMMISSIONING_MONITOR"
                if self.commissioning_hall_monitor_only
                else "SYNCHRONIZED_CAPTURE"
                if self.require_force
                else "DISABLED"
            ),
            "synchronized": bool(
                self.require_force and self._force_sync_metadata is not None
            ),
            "training_eligible": False,
            "hardware": copy.deepcopy(self._force_hardware),
            "status": self._force_status,
            "ready": False,
            "recording": self._force_recording_active,
            "force_n": None,
            "hard_limit_force_n": self._force_hard_limit_n,
            "age_s": None,
            "error": self._force_error,
        }
        if self._force is None:
            if self._force_requested:
                base.update(status="MISSING", ready=False)
            return base
        try:
            sample = self._force.read()
            status = str(sample.status)
            ready = status in {"OK", "WARN"}
            base.update({
                "status": status,
                "ready": ready,
                "force_n": round(float(sample.force_n), 3),
                "age_s": float(getattr(sample, "age_s", 0.0)),
                "displacement_mm": float(getattr(sample, "displacement_mm", 0.0)),
                "raw": int(getattr(sample, "raw", 0)),
                "hall_v": float(getattr(sample, "hall_v", 0.0)),
                "device_t_ms": getattr(sample, "device_t_ms", None),
                "error": None if ready else self._force_error,
                # Force is ADVISORY only: it must never assert or influence training
                # eligibility. Only the finalizer promotes, and no producer surface may
                # emit training_eligible=true. A valid over-threshold physician reading is
                # NOT a downgrade either -- keep this a constant False, decoupled from the
                # force status entirely.
                "training_eligible": False,
            })
            if (
                self.require_force
                and self.platform.state is CollectionState.RECORDING
                and not ready
            ):
                # P0-4: the Hall force reading is a physician-facing advisory reference,
                # NOT a safety interlock. A high/low/over-threshold reading during
                # recording updates the displayed status only -- it NEVER aborts
                # recording and never touches the arm.
                self._force_status = status
                self._force_error = f"Hall-force {status} during recording (advisory)"
        except Exception as exc:  # noqa: BLE001 - advisory health readout
            base.update({
                "status": "ERROR", "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            # Advisory: a Hall read failure degrades the reference channel; it does not
            # abort recording (_on_force_capture_fault is itself advisory now).
            if self.platform.state is CollectionState.RECORDING:
                self._on_force_capture_fault(exc)
        return base

    def audio_monitor_chunk(self, cursor, max_samples):
        if self._audio is None:
            raise AttributeError("live audio is disabled")
        return self._audio.monitor_chunk(cursor, max_samples=max_samples)

    def _capture_target_dir(self):
        target = Path(self.out_dir)
        while not target.exists() and target != target.parent:
            target = target.parent
        return target

    def _estimated_capture_bytes_per_s(self):
        """Conservative on-disk byte rate for a diagnostic session.

        Depth is stored losslessly zlib-compressed (~14x typical); a conservative
        ratio is applied to the aligned raw u16 depth rate.  RGB is mp4 video, so a
        conservative non-depth allowance covers RGB + audio + force + robot ledger +
        tracker sidecars.  Returns 0.0 if the camera mode is not fully pinned.
        """
        try:
            depth_raw = (
                int(self.camera_width) * int(self.camera_height)
                * np.dtype(np.uint16).itemsize * float(self.depth_fps)
            )
        except (TypeError, ValueError):
            return 0.0
        if depth_raw <= 0:
            return 0.0
        depth = depth_raw / self.depth_compression_ratio
        return (depth + self.non_depth_capture_bytes_per_s) * float(self.capture_rate_margin)

    def _storage_start_rejection(self, planned_duration_s):
        """Refuse Start unless the disk holds the whole planned session + reserve.

        Returns a human-readable rejection string, or None when there is room (or
        when the gate is disabled / the rate cannot be estimated).
        """
        try:
            planned = float(planned_duration_s)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(planned) or planned <= 0:
            return None  # gate disabled
        rate = self._estimated_capture_bytes_per_s()
        if rate <= 0:
            return None  # cannot estimate; other camera-mode gates cover this
        free = int(shutil.disk_usage(self._capture_target_dir()).free)
        reserve = int(self.diagnostic_min_free_bytes)
        usable = max(0, free - reserve)
        required = int(planned * rate)
        if usable < required:
            recordable_s = usable / rate
            return (
                f"insufficient storage for a {planned:.0f}s session: only "
                f"~{recordable_s:.0f}s recordable above the "
                f"{reserve / 1024 ** 3:.0f} GiB finalization reserve "
                f"(free={free / 1024 ** 3:.1f} GiB, "
                f"rate={rate / 1024 ** 2:.0f} MiB/s)"
            )
        return None

    def _idle_storage_health(self):
        """Estimate safe recording time before a session capture exists."""

        target = Path(self.out_dir)
        while not target.exists() and target != target.parent:
            target = target.parent
        free = int(shutil.disk_usage(target).free)
        usable = max(0, free - self.diagnostic_min_free_bytes)
        try:
            # DiagnosticPerceptionCapture persists depth ALIGNED to the RGB frame
            # (perception_capture requires depth.shape == the color frame shape),
            # so on-disk depth is camera_width x camera_height (e.g. 1920x1080),
            # NOT the raw depth-mode size (e.g. 640x576). Estimating with the
            # depth-mode size under-counts the depth byte rate ~5.6x and grossly
            # overstates the recordable time.
            depth_rate = (
                int(self.camera_width) * int(self.camera_height) *
                np.dtype(np.uint16).itemsize * float(self.depth_fps)
            )
        except (TypeError, ValueError):
            depth_rate = 0.0
        remaining = None if depth_rate <= 0 else usable / depth_rate
        return {
            "status": "OK" if usable > 0 else "RESERVE_EXHAUSTED",
            "depth_format": depth_codec.DEPTH_STORAGE_ID,
            "free_bytes": free,
            "safety_watermark_bytes": self.diagnostic_min_free_bytes,
            "final_rgb_copy_reserve_bytes": 0,
            "usable_bytes_before_safety_reserve": usable,
            "persisted_depth_bytes": 0,
            "persisted_capture_bytes_estimate": 0,
            "expected_raw_depth_bytes_per_second": depth_rate,
            "measured_capture_bytes_per_second": 0.0,
            "conservative_bytes_per_second": depth_rate,
            "estimated_remaining_seconds": remaining,
        }

    def preview_snapshot(self):
        """Return one atomic JPEG/index pair for browser-visible frame binding."""

        with self._lock:
            jpeg = bytes(self.jpeg)
            confirmable = bool(self._preview_frame_confirmable)
            return {
                "jpeg": jpeg,
                "frame_index": int(self._preview_frame_index),
                "frame_host_t_s": self._preview_frame_t,
                "generation": int(self._preview_generation),
                "confirmable": confirmable,
                "frame_token": (
                    self._preview_frame_token(self._preview_frame_index, jpeg)
                    if confirmable else None
                ),
            }

    def _preview_frame_token(self, frame_index, jpeg):
        jpeg_sha256 = hashlib.sha256(bytes(jpeg)).hexdigest()
        signature = hmac.new(
            self._preview_token_key,
            (
                "wam-preview-frame-v3:"
                f"{int(self._preview_generation)}:{int(frame_index)}:{jpeg_sha256}"
            ).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{jpeg_sha256}.{signature}"

    def _preview_frame_token_valid(self, frame_index, token):
        with self._lock:
            if (
                not self._preview_frame_confirmable
                or not isinstance(token, str)
                or len(token) != 129
                or token[64] != "."
            ):
                return False
            jpeg_sha256, signature = token.split(".", 1)
            if (
                re.fullmatch(r"[0-9a-f]{64}", jpeg_sha256) is None
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None
            ):
                return False
            expected = hmac.new(
                self._preview_token_key,
                (
                    "wam-preview-frame-v3:"
                    f"{int(self._preview_generation)}:"
                    f"{int(frame_index)}:{jpeg_sha256}"
                ).encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(signature, expected)

    def state_snapshot(self):
        snapshot = self.platform.state_snapshot()
        with self._lock:
            preview_index = self._preview_frame_index
            preview_t = self._preview_frame_t
            preview_generation = self._preview_generation
            preview_publish_times = tuple(self._preview_publish_times)
            fatal_error = self._fatal_error
            current_tracking = copy.deepcopy(self._current_tracking)
        now = time.monotonic()
        if (
            self._record_started is not None
            and self._record_stopped is None
            and snapshot["state"] != CollectionState.RECORDING.value
        ):
            self._record_stopped = now
        record_end = self._record_stopped or now
        elapsed = (
            0.0 if self._record_started is None
            else max(0.0, record_end - self._record_started)
        )
        model_health = (
            self._safe_worker_health(self._model_worker, label="model")
            if self._model_worker is not None
            else {"status": self._model_status, "error": fatal_error}
        )
        ingress_health = self._safe_worker_health(
            self._ingress_worker, label="capture ingress",
        )
        camera_delivery_health = self._camera_delivery_health()
        preview_health = self._safe_worker_health(
            self._preview_worker, label="preview",
        )
        capture_storage = None
        storage_getter = getattr(self.platform.capture, "storage_health", None)
        if callable(storage_getter):
            try:
                capture_storage = storage_getter()
            except Exception as exc:  # noqa: BLE001 - health is evidence, not success
                capture_storage = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if capture_storage is None:
            try:
                capture_storage = self._idle_storage_health()
            except OSError as exc:
                capture_storage = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        preview_produced_fps = 0.0
        if len(preview_publish_times) >= 2:
            span = preview_publish_times[-1] - preview_publish_times[0]
            if span > 0:
                preview_produced_fps = (
                    len(preview_publish_times) - 1
                ) / span
        runtime_health = self._runtime_health_snapshot(now=now)
        robot = runtime_health.get("robot", self._robot_snapshot())
        force = (
            runtime_health["force"]
            if "force" in runtime_health else self._force_snapshot()
        )
        audio = self._audio_snapshot()
        back_confirmation = self._back_confirmation_snapshot()
        perception_stages = perception_stage_snapshot(
            model_status=self._model_status,
            tracking=current_tracking,
            back_confirmation=back_confirmation,
            model_release_id=self.model_release_id,
        )
        runtime_strict_ok = runtime_health["strict_ok"] is True
        points = (
            visible_tracking_points(current_tracking, expected_names=self.point_names)
            if runtime_strict_ok else []
        )
        active_name = snapshot["point"]
        target_valid = runtime_strict_ok and bool(
            current_tracking.get("constellation_valid")
        ) and any(
            point.get("name") == active_name for point in points
        )
        model_result_current = (
            self.fake
            or (
                model_health.get("status") == "OK"
                and model_health.get("last_result_age_s") is not None
                and model_health["last_result_age_s"] <= self.max_tracking_age_s
            )
        )
        diagnostic_ready = (
            runtime_strict_ok
            and self._camera_status == "OK"
            and self._model_status == "RUNNING"
            and model_result_current
            and (
                not camera_delivery_health.get("interval_active")
                or camera_delivery_health.get("status") == "OK"
            )
            and fatal_error is None
            and not snapshot["diagnostic_blockers"]
            and (not self.require_robot or robot.get("ready") is True)
            and (not self._force_requested or force.get("ready") is True)
            and (
                (self.audio_device is None and not self.require_audio)
                or (
                    audio["status"] == "OK"
                    and (not self.require_audio or audio["signal_present"] is True)
                )
            )
        )
        finalization_summary = None
        finalization = snapshot.get("finalization")
        if (
            snapshot["state"] == CollectionState.COMPLETE.value
            and isinstance(finalization, dict)
            and snapshot.get("session_dir")
        ):
            finalization_summary = self._verified_finalization_summary(
                snapshot, finalization,
            )
        # The complete seal and artifact inventory remain on disk. Never return
        # or retain their segment-count-sized copy in the hot /state payload.
        snapshot["finalization"] = None
        if snapshot["state"] == CollectionState.RECORDING.value:
            session_frames = int(ingress_health.get("accepted", 0) or 0)
        elif finalization_summary and finalization_summary.get("complete"):
            session_frames = int(finalization_summary.get("n_rgb_frames", 0) or 0)
        else:
            session_frames = 0
        snapshot.update({
            "idx": snapshot["point_index"],
            "n": len(self.point_names),
            "names": list(self.point_names),
            "recording": snapshot["state"] == CollectionState.RECORDING.value,
            "elapsed": round(elapsed, 1),
            "status": snapshot["error"] or snapshot["action_ack"] or current_tracking.get("reason"),
            "feedback_state": current_tracking.get("reason"),
            "force": (
                float(force["force_n"])
                if isinstance(force.get("force_n"), (int, float))
                and not isinstance(force.get("force_n"), bool)
                and math.isfinite(float(force["force_n"]))
                else None
            ),
            "cycles": 0,
            "need": 2,
            "occluded": any(point["display_status"] == "coasting" for point in points),
            "points_visible": len(points),
            "points": points,
            "back_present": bool(
                runtime_strict_ok and current_tracking.get("back_present", False)
            ),
            "initialized": bool(
                runtime_strict_ok and current_tracking.get("initialized", False)
            ),
            "target_valid": target_valid,
            "tracking_reason": str(current_tracking.get("reason", "MODEL_STARTING")),
            "posterior_gate": current_tracking.get("posterior_gate"),
            "back_confirmation": back_confirmation,
            "perception_stages": perception_stages,
            "tracking_age_s": current_tracking.get("tracking_age_s"),
            "preview_generation": preview_generation,
            "preview_frame_index": preview_index,
            "preview_age_s": None if preview_t is None else max(0.0, now - preview_t),
            "camera_status": self._camera_status,
            "model_status": self._model_status,
            "model_health": model_health,
            "capture_ingress": ingress_health,
            "camera_delivery": camera_delivery_health,
            "preview_health": preview_health,
            "preview_produced_fps": preview_produced_fps,
            "session_recorded_frames": session_frames,
            "prestart_frames_saved": False,
            "audio": audio,
            "force_sensor": force,
            "robot": robot,
            "controller": self._controller_status_payload(),
            "handeye": self._handeye_snapshot(),
            "capture_storage": capture_storage,
            "fatal_error": fatal_error,
            "diagnostic_ready": diagnostic_ready,
            "commissioning_preview_only": self.commissioning_preview_only,
            "recording_controls_enabled": not self.commissioning_preview_only,
            "capture_mode": (
                "COMMISSIONING_PREVIEW_ONLY"
                if self.commissioning_preview_only else "DIAGNOSTIC_CAPTURE"
            ),
            "runtime_health": runtime_health,
            "finalization_summary": finalization_summary,
        })
        # Unified Start authority.  The page consumes ONLY snapshot.start_authorized,
        # so it must fold the platform's decision together with the collector-owned
        # hardware gates (camera/model readiness, live stethoscope audio, back
        # confirmation) the platform cannot see.  These mirror the HTTP
        # _control_locked start checks exactly, so the page and API never disagree.
        platform_ok = snapshot.get("start_authorized") is True
        # Device readiness is the Start precondition: the camera and model must be
        # up, with no fatal error or outstanding preflight blocker.  The richer
        # per-frame recording-quality signals (live constellation strictness,
        # delivery-interval health) are NOT Start gates -- they govern an active
        # recording and the back-confirmation handshake, not whether Start may
        # begin -- so folding them in here would wrongly forbid a legitimate
        # Start (and diverge from the operator's confirm-then-start flow).
        device_ready = self._device_start_rejection() is None
        audio_gate_ok = True
        audio_reason = None
        if self.require_audio or self.audio_device is not None:
            if audio.get("status") != "OK":
                audio_gate_ok = False
                audio_reason = (
                    "stethoscope audio stream is not live: %s" % audio.get("status")
                )
            elif self.require_audio and audio.get("signal_present") is not True:
                audio_gate_ok = False
                audio_reason = "no stethoscope signal is present"
        back_gate_ok = not (
            back_confirmation.get("required") is True
            and back_confirmation.get("confirmed") is not True
        )
        unified_ok = (
            platform_ok and device_ready and audio_gate_ok and back_gate_ok
        )
        if unified_ok:
            unified_reason = None
        elif not platform_ok:
            unified_reason = snapshot.get("start_rejection") or "start is not authorized"
        elif not device_ready:
            unified_reason = "camera/model preflight is not ready"
        elif not audio_gate_ok:
            unified_reason = audio_reason
        else:
            unified_reason = "confirm the patient's back before starting"
        snapshot["start_authorized"] = unified_ok
        snapshot["start_rejection"] = unified_reason
        return snapshot


def make_handler(col: Collector, page: bytes):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            request = urlsplit(self.path)
            path = request.path
            if path == "/":
                self._send(page, "text/html; charset=utf-8", no_store=True)
            elif path == "/state":
                self._send(
                    json.dumps(_finite_json(col.state_snapshot())).encode(), "application/json",
                    no_store=True,
                )
            elif path == "/healthz":
                state = col.state_snapshot()
                healthy = state.get("diagnostic_ready") is True
                self._send(
                    json.dumps({"ok": healthy, "camera": state["camera_status"],
                                "model": state["model_status"]}).encode(),
                    "application/json", status=200 if healthy else 503, no_store=True,
                )
            elif path == "/frame.jpg":
                frame = col.preview_snapshot()
                query = parse_qs(request.query, keep_blank_values=True)
                try:
                    requested_after = int(query.get("after", [""])[-1])
                    requested_generation = int(
                        query.get("generation", [""])[-1]
                    )
                except (TypeError, ValueError):
                    requested_after = requested_generation = None
                if (
                    requested_after == int(frame["frame_index"])
                    and requested_generation == int(frame["generation"])
                ):
                    self._send(
                        b"", "image/jpeg", status=204, no_store=True,
                    )
                    return
                jpeg = frame["jpeg"]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("X-Preview-Frame-Index", str(int(frame["frame_index"])))
                self.send_header("X-Preview-Generation", str(int(frame["generation"])))
                self.send_header(
                    "X-Preview-Confirmable",
                    "1" if frame["confirmable"] else "0",
                )
                if frame["frame_token"] is not None:
                    self.send_header(
                        "X-Preview-Frame-Token", str(frame["frame_token"]),
                    )
                if frame.get("frame_host_t_s") is not None:
                    self.send_header(
                        "X-Preview-Frame-Host-T",
                        str(float(frame["frame_host_t_s"])),
                    )
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif path == "/stream":
                self._send(
                    b'{"ok":false,"message":"obsolete unbounded stream endpoint"}',
                    "application/json", status=410, no_store=True,
                )
            elif path == "/audio/pcm":
                try:
                    query = parse_qs(request.query, keep_blank_values=True)
                    cursor_values = query.get("cursor")
                    cursor = None if not cursor_values else int(cursor_values[-1])
                    requested = int(query.get("samples", [4_410])[-1])
                    max_samples = max(1, min(requested, 22_050))
                    chunk = col.audio_monitor_chunk(cursor, max_samples)
                    samples = np.asarray(chunk["samples"], dtype=np.float32)
                    if samples.ndim != 1:
                        raise RuntimeError("audio monitor returned non-mono samples")
                    pcm = (
                        np.clip(samples, -1.0, 1.0) * 32767.0
                    ).astype("<i2").tobytes()
                except (AttributeError, KeyError):
                    self._send(
                        b'{"ok":false,"message":"live audio is disabled"}',
                        "application/json", status=503, no_store=True,
                    )
                    return
                except (TypeError, ValueError, RuntimeError) as exc:
                    self._send(
                        json.dumps({"ok": False, "message": str(exc)}).encode(),
                        "application/json", status=400, no_store=True,
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Audio-Encoding", "pcm_s16le_mono")
                self.send_header("X-Audio-Sample-Rate", str(int(chunk["sample_rate"])))
                self.send_header("X-Audio-Start-Sample", str(int(chunk["start_sample"])))
                self.send_header("X-Audio-Next-Sample", str(int(chunk["next_sample"])))
                self.send_header("X-Audio-Overrun", "1" if chunk["overrun"] else "0")
                self.send_header("X-Audio-Status", str(chunk["status"]))
                self.send_header("Content-Length", str(len(pcm)))
                self.end_headers()
                self.wfile.write(pcm)
            else:
                self._send(
                    b'{"ok":false,"message":"not found"}',
                    "application/json", status=404, no_store=True,
                )

        def do_POST(self):
            if self.path == "/control":
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    self._send(b'{"ok":false,"message":"Content-Length required"}',
                               "application/json", status=411, no_store=True)
                    return
                try:
                    length = int(raw_length)
                except ValueError:
                    self._send(b'{"ok":false,"message":"invalid Content-Length"}',
                               "application/json", status=400, no_store=True)
                    return
                if length < 0 or length > 16 * 1024:
                    self._send(b'{"ok":false,"message":"request body too large"}',
                               "application/json", status=413, no_store=True)
                    return
                try:
                    body = json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send(b'{"ok":false,"message":"malformed JSON"}',
                               "application/json", status=400, no_store=True)
                    return
                if not isinstance(body, dict) or "action" not in body:
                    self._send(b'{"ok":false,"message":"body requires action"}',
                               "application/json", status=400, no_store=True)
                    return
                action = body.get("action")
                if action == "arm_disable":
                    # E-STOP FAST PATH: the console disable MUST NOT queue behind an
                    # in-flight bring-up/preflight/readiness-wait that holds _control_lock.
                    # It bypasses _control_lock entirely and goes straight to the
                    # controller companion (guarded only by the short-held _controller_lock,
                    # which bring-up releases for the duration of its launch/await). Without
                    # this, an E-stop pressed during a bring-up would not disable the arm
                    # until the bring-up finished.
                    if set(body) != {"action"}:
                        self._send(
                            b'{"ok":false,"message":"body requires only action"}',
                            "application/json", status=400, no_store=True,
                        )
                        return
                    result = col.disable_controller()
                elif action == "confirm_back":
                    if (
                        set(body) != {
                            "action", "preview_frame_index", "preview_frame_token",
                        }
                        or type(body.get("preview_frame_index")) is not int
                        or not isinstance(body.get("preview_frame_token"), str)
                    ):
                        self._send(
                            b'{"ok":false,"message":"Confirm Back requires preview_frame_index"}',
                            "application/json", status=400, no_store=True,
                        )
                        return
                    result = col.control(
                        action,
                        preview_frame_index=body["preview_frame_index"],
                        preview_frame_token=body["preview_frame_token"],
                    )
                else:
                    if set(body) != {"action"}:
                        self._send(
                            b'{"ok":false,"message":"body requires only action"}',
                            "application/json", status=400, no_store=True,
                        )
                        return
                    result = col.control(action)
                status = 200 if result.get("ok") else (
                    400 if result.get("code") == "invalid_action" else 409
                )
                self._send(
                    json.dumps(result).encode(), "application/json",
                    status=status, no_store=True,
                )
            else:
                self._send(
                    b'{"ok":false,"message":"not found"}',
                    "application/json", status=404, no_store=True,
                )

        def _send(self, b, ct, *, status=200, no_store=False):
            self.send_response(status); self.send_header("Content-Type", ct)
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    return H


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auscultation Capture</title>
<style>
:root{--bg:#070a0d;--panel:#10151a;--line:#283039;--ink:#edf1f4;--muted:#96a0a9;--dim:#5f6871;
--cyan:#62d7f5;--amber:#ffc857;--green:#5ee6a8;--red:#ff667d;--mono:"SFMono-Regular",Menlo,Consolas,monospace;
--sans:"Avenir Next",Avenir,"Helvetica Neue",sans-serif}*{box-sizing:border-box}body{margin:0;min-height:100vh;
background:linear-gradient(115deg,#070a0d 0,#0b1014 54%,#070a0d 100%);color:var(--ink);font-family:var(--sans)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.17;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:28px 28px}
.shell{position:relative;max-width:1440px;margin:auto;padding:20px}.mast{display:flex;justify-content:space-between;gap:18px;
align-items:flex-end;padding:7px 0 16px;border-bottom:1px solid var(--line)}.eyebrow,.label{font:700 10px/1.3 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
h1{font:650 clamp(21px,2.4vw,34px)/1.05 var(--sans);margin:7px 0 0}.scope{max-width:480px;border-left:3px solid var(--red);padding:8px 12px;background:#211218}
.scope strong{display:block;color:var(--red);font:800 12px/1.3 var(--mono);letter-spacing:.08em}.scope span{display:block;color:#dfb2ba;font-size:12px;margin-top:3px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:16px;margin-top:16px}@media(max-width:980px){.layout{grid-template-columns:1fr}.shell{padding:12px}.mast{align-items:flex-start;flex-direction:column}}
.stage{min-width:0}.feed{position:relative;aspect-ratio:16/9;background:#000;border:1px solid var(--line);overflow:hidden;box-shadow:0 24px 60px #0008}
.feed canvas{width:100%;height:100%;object-fit:contain;display:block}.feedhead{position:absolute;inset:0 0 auto;display:flex;gap:12px;justify-content:space-between;padding:10px 12px;background:linear-gradient(#050709e8,transparent);font:700 10px var(--mono);letter-spacing:.1em;text-transform:uppercase}
.live{display:flex;align-items:center;gap:7px}.live i{width:7px;height:7px;border-radius:50%;background:var(--dim)}.live.on i{background:var(--red);animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.25}}
.feedfoot{position:absolute;inset:auto 0 0;padding:12px;background:linear-gradient(transparent,#050709eb 36%)}
#statusbar{font:650 14px/1.35 var(--mono);text-shadow:0 1px 2px #000}.telemetry{display:flex;gap:18px;margin-top:7px;color:var(--muted);font:11px var(--mono);flex-wrap:wrap}
.action{min-height:48px;margin-top:12px;padding:12px 14px;border:1px solid var(--line);background:#0c1115;font:12px/1.45 var(--mono);color:var(--muted)}.action.good{border-color:#235e49;color:var(--green)}.action.bad{border-color:#71303d;color:var(--red)}
.rail{display:flex;flex-direction:column;gap:12px}.card{background:linear-gradient(145deg,#12181e,#0d1216);border:1px solid var(--line);padding:14px}.railcard{min-width:0;padding:12px}.cardhead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}
.pill{font:800 10px var(--mono);letter-spacing:.08em;border:1px solid currentColor;padding:5px 8px}.target{font:750 34px/1 var(--mono);letter-spacing:-.05em}.meta{display:grid;grid-template-columns:100px 1fr;gap:7px 10px;margin-top:13px;font:11px/1.35 var(--mono)}.meta dt{color:var(--dim)}.meta dd{margin:0;overflow-wrap:anywhere}
.back{margin-top:13px;padding:9px 10px;border-left:3px solid var(--red);background:#190f13;color:#e0a7b0;font:700 11px/1.4 var(--mono)}.back.ok{border-color:var(--green);background:#0b1914;color:var(--green)}
.points{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px 4px}.point{display:flex;align-items:center;gap:7px;min-width:0;padding:4px 6px;color:var(--muted);font:10px/1.2 var(--mono);border:1px solid transparent}.point span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.point i{width:7px;height:7px;flex:0 0 auto;border:1px solid var(--dim);border-radius:50%}.point.active{color:var(--amber);border-color:#594825;background:#1a160d}.point.active i{background:var(--amber);border-color:var(--amber)}.point.done{color:var(--green)}.point.done i{background:var(--green);border-color:var(--green)}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.backcontrol{margin-top:8px}button{appearance:none;border:1px solid #3b4650;background:#161d23;color:var(--ink);min-height:42px;padding:9px;font:750 11px var(--mono);cursor:pointer}button:hover:not(:disabled){border-color:var(--cyan);color:var(--cyan)}button:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}button:disabled{opacity:.35;cursor:not-allowed}.start{border-color:#28634d;color:var(--green)}.end{border-color:#7b3340;color:var(--red)}.wide{grid-column:1/-1}.estop{border-color:var(--red);color:var(--red);font-weight:800}.estop:hover:not(:disabled){background:var(--red);color:#0a0d10;border-color:var(--red)}.armctl{margin-bottom:10px;padding:8px;border:1px solid #3b4650;border-radius:4px}.armstat{font:750 11px var(--mono);margin-bottom:6px}.armstat.good{color:var(--green)}.armstat.bad{color:var(--red)}.armstat.warn{color:#e0a030}
.keyline{margin-top:10px;color:var(--dim);font:10px/1.5 var(--mono)}.blockers{margin:0;padding-left:17px;color:#d2a5ad;font:11px/1.45 var(--mono)}.blockers li+li{margin-top:6px}.sessionpath{font:10px/1.4 var(--mono);color:var(--muted);overflow-wrap:anywhere}
.capturestrip{margin-top:12px;padding:11px 12px;border:1px solid var(--line);background:linear-gradient(145deg,#11171c,#0b1014)}.capturetop{display:flex;align-items:center;justify-content:space-between;gap:12px}.capturemetrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:9px}.capturemetric{min-width:0;padding-left:9px;border-left:2px solid #26313a}.capturemetric b{display:block;margin-top:4px;font:750 14px/1.2 var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.capturefoot{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;margin-top:9px}.capturefoot>:last-child{text-align:right}@media(max-width:720px){.capturemetrics{grid-template-columns:repeat(2,minmax(0,1fr))}.capturefoot{grid-template-columns:1fr}.capturefoot>:last-child{text-align:left}}
.audiolevel{height:8px;border-radius:10px;background:#080b0e;overflow:hidden;border:1px solid var(--line);margin:10px 0}.audiolevel i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green),var(--amber));transition:width .18s ease}
.audio-row{display:flex;align-items:center;justify-content:space-between;gap:10px}.audio-copy{font:11px/1.45 var(--mono);color:var(--muted)}.audio-copy strong{display:block;color:var(--ink);font-size:12px}.listen{min-width:112px;color:var(--cyan);border-color:#285967}.listen.on{color:var(--red);border-color:#7b3340}
.sensorblock+.sensorblock{border-top:1px solid var(--line);margin-top:10px;padding-top:10px}.sensorblock .cardhead{margin-bottom:7px}.sensorblock .keyline{margin-top:7px}.forcelevel{position:relative;height:12px;border-radius:10px;background:#080b0e;overflow:hidden;border:1px solid var(--line);margin:8px 0}.forcelevel:after{content:"";position:absolute;inset:0 auto 0 83.64%;width:1px;background:#8b7136}.forcelevel i{display:block;height:100%;width:0;background:var(--dim);transition:width .18s ease}.taskpoints{margin-top:10px;padding-top:9px;border-top:1px solid var(--line)}.taskpoints .cardhead{margin-bottom:5px}#taskcard .back{margin-top:10px;padding:7px 9px}#taskcard .backcontrol{margin-top:7px}#taskcard>.keyline{margin-top:7px}#controlcard .cardhead{margin-bottom:9px}#controlcard .buttons{gap:7px}#controlcard button{min-height:40px}
details.engineering{margin-top:10px;background:#0c1115;border:1px solid var(--line);padding:10px 12px}details.engineering summary{cursor:pointer;color:var(--muted);font:700 10px var(--mono);letter-spacing:.08em}details.engineering .telemetry{margin-top:10px}details.engineering .blockerhead{display:block;margin:12px 0 7px}details.engineering .blockers{font-size:10px}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body><main class="shell">
<header class="mast"><div><div class="eyebrow">LINBOX · LIVE CAPTURE</div><h1>Auscultation Capture</h1></div>
<div class="scope"><strong>COMMISSIONING</strong><span>NOT TRAINING-READY</span></div></header>
<section class="layout"><div class="stage"><div class="feed" id="feed"><canvas id="camera" role="img" aria-label="Live Orbbec camera preview">Live camera preview</canvas>
<div class="feedhead"><span>Orbbec · Live tracking</span><span class="live" id="live"><i></i><b id="recword">Idle</b></span></div>
<div class="feedfoot"><div id="statusbar">Connecting…</div><div class="telemetry"><span id="livefps">Live FPS —</span><span id="sessionframes">Session frames 0</span><span id="pointstat">Points 0 / 10</span></div></div></div>
<div class="action" id="actionmsg" role="status" aria-live="polite">Loading…</div>
<section class="capturestrip" id="capturestrip"><div class="capturetop"><span class="label">Capture</span><span class="pill" id="statepill">—</span></div><div class="capturemetrics">
<div class="capturemetric"><span class="label">Duration</span><b id="elapsed">0.0 s</b></div><div class="capturemetric"><span class="label">Written</span><b id="framecount">0 frames</b></div><div class="capturemetric"><span class="label">Point</span><b id="attempt">—</b></div><div class="capturemetric"><span class="label">Storage</span><b id="storage">Estimating</b></div></div>
<div class="capturefoot"><div><div class="keyline" id="capturehint">Start when ready; end to save and verify.</div><div class="sessionpath" id="sessionpath">No session</div></div><div class="sessionpath" id="videoresult">Verified on end</div></div></section>
<details class="engineering"><summary>System details</summary><div class="telemetry"><span id="camstat">Camera —</span><span id="modelstat">Model —</span><span id="robotstat">Robot —</span><span id="handeyestat">Hand-eye —</span><span id="dropstat">Drops —</span></div><span class="label blockerhead">Training gates</span><ul class="blockers" id="blockers"><li>Loading…</li></ul></details></div>
<aside class="rail"><section class="card railcard" id="taskcard"><div class="cardhead"><span class="label">Current point</span><span id="counter" class="label">—</span></div><div class="target" id="tname">—</div><div class="back" id="backgate">Waiting for back</div><div class="buttons backcontrol"><button class="wide" id="btnback">Confirm Back</button></div>
<div class="taskpoints"><div class="cardhead"><span class="label">10 auscultation points</span><span id="pointslabel" class="label">0 / 10</span></div><div class="points" id="plist"></div></div><div class="keyline">Face away, stay alone in frame, then Confirm Back.</div></section>
<section class="card railcard" id="sensorcard"><div class="sensorblock"><div class="cardhead"><span class="label">Lung audio</span><span class="pill" id="audiopill">—</span></div><div class="audio-row"><div class="audio-copy"><strong id="audiostatus">Checking input</strong><span id="audiodb">— dBFS</span></div><button class="listen" id="btnaudio">▶ Monitor</button></div><div class="audiolevel" aria-hidden="true"><i id="audiofill"></i></div><div class="keyline">Headphones required.</div></div>
<div class="sensorblock"><div class="cardhead"><span class="label">Contact force</span><span class="pill" id="forcepill">—</span></div><div class="forcelevel" id="forcebar" role="progressbar" aria-label="Contact force level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-valuetext="Unavailable"><i id="forcefill"></i></div><div class="keyline">Physician reference only — never stops recording or the arm.</div></div></section>
<section class="card railcard" id="controlcard"><div class="cardhead"><span class="label">Controls</span></div><div class="buttons">
<div class="armctl" id="armctl" hidden><div class="armstat" id="armstat">Arm —</div><div class="buttons"><button class="wide" id="btnarm">⬆ Bring up arm</button><button class="wide estop" id="btnestop">■ Disable (E-stop)</button></div></div>
<button class="start wide" id="btnstart">● Start</button><button class="end wide" id="btnend">■ End &amp; Verify</button>
<button id="btnconfirm">✓ Confirm</button><button id="btnredo">↻ Redo</button><button id="btnprev">← Previous</button><button id="btnnext">Next →</button></div>
</section></aside></section></main>
<script>
const $=id=>document.getElementById(id);let latest=null,busy=false,lastActionAck=null;
let displayedPreviewFrameIndex=null,displayedPreviewFrameToken=null,lastRenderedFrameIndex=null,lastRenderedGeneration=null,cameraTimer=null,cameraInFlight=false,pollTimer=null,pollInFlight=false;
let listening=false,audioContext=null,audioCursor=null,audioNextAt=0,audioTimer=null;
let transportDown=false,lastBlockersKey=null,pageStopped=false;
function action(text,kind=''){const el=$('actionmsg');el.textContent=text;el.className='action'+(kind?' '+kind:'')}
function fmtAge(v){return v==null?'—':Number(v).toFixed(2)+'s'}
function fmtBytes(v){if(v==null||!Number.isFinite(Number(v)))return '—';let n=Math.max(0,Number(v)),u='B';for(const next of ['KiB','MiB','GiB','TiB']){if(n<1024)break;n/=1024;u=next}return n.toFixed(n>=100?0:n>=10?1:2)+' '+u}
function fmtRemaining(v){if(v==null||!Number.isFinite(Number(v)))return 'Estimating';const seconds=Math.max(0,Number(v));if(seconds<120)return '~ '+Math.floor(seconds)+' s';if(seconds<7200)return '~ '+(seconds/60).toFixed(1)+' min';return '~ '+(seconds/3600).toFixed(1)+' hr'}
const stateText={READY:'Ready',RECORDING:'Recording',FINALIZING:'Verifying',COMPLETE:'Complete',PREFLIGHT_FAILED:'Not ready',ERROR:'Error',ABORTED:'Stopped',DISABLED:'Off',WAITING:'Waiting'};
const reasonText={NO_PERSON:'Waiting for person',AMBIGUOUS_PEOPLE:'One person only',PERSON_DETECTION_INVALID:'Person detector invalid',BACK_CONFIRMATION_REQUIRED:'Confirm patient back',BACK_CONFIRMATION_PENDING_FRAME:'Checking person',BACK_NOT_CONFIRMABLE:'Waiting for one person',PREVIEW_FRAME_NOT_BOUND:'Preview changed · confirm again',OPERATOR_CONFIRMED_BACK:'Back confirmed',CONFIRMATION_TARGET_LOST:'Person lost · confirm again',CONFIRMATION_OBSERVATION_STALE:'Person feed stale · confirm again',PERSON_OBSERVATION_STALE:'Waiting for current person',CONFIRMED_PERSON_CHANGED:'Person changed · confirm again',NON_MONOTONIC_PERSON_FRAME:'Person feed reset · confirm again',PERSON_GATE_STARTING:'Starting person detector',NO_BACK:'Waiting for back',MODEL_STARTING:'Starting model',TRACKING:'Tracking',REACQUIRING:'Reacquiring',INITIALIZING:'Initializing'};
const blockerText={'physical audio-force sync calibration is missing':'Audio-force sync','Orbbec mode-bound hardware LED latency calibration is missing':'Camera latency','camera-to-robot hand-eye calibration and robot-base targets are missing':'Hand-eye calibration','confirmed direct motor-disable command is missing':'Motor stop','live robot/action, Hall force, and stethoscope audio identities are not commissioned':'Hardware identities'};
const ackText={'Diagnostic preflight ready':'Ready to start','Diagnostic recording started':'Recording started','End accepted; diagnostic recording is finalizing':'Ending · verifying','Diagnostic recording saved and verified (not training eligible)':'Saved · verified'};
function shortText(v){return ackText[v]||reasonText[v]||stateText[v]||String(v||'—').replaceAll('_',' ')}
function pointList(s){const root=$('plist');if(root.children.length!==s.n){root.textContent='';s.names.forEach((name,index)=>{const row=document.createElement('div');row.className='point';row.id='p'+index;const dot=document.createElement('i');const label=document.createElement('span');label.textContent=name;row.append(dot,label);root.append(row)})}s.names.forEach((_,index)=>{$('p'+index).className='point'+(s.completed.includes(index)?' done':index===s.idx?' active':'')})}
function blockerList(items){const labels=(items.length?items:['None']).map(item=>blockerText[item]||item),key=JSON.stringify(labels);if(key===lastBlockersKey)return;lastBlockersKey=key;const root=$('blockers');root.textContent='';labels.forEach(label=>{const li=document.createElement('li');li.textContent=label;root.append(li)})}
function syncBackButton(back=latest?.back_confirmation||{}){const button=$('btnback');button.disabled=busy||back.required!==true||back.confirmable!==true||back.active===true||!Number.isInteger(displayedPreviewFrameIndex)||typeof displayedPreviewFrameToken!=='string';button.textContent=back.confirmed?'Back Confirmed':back.active?'Checking…':'Confirm Back'}
function render(s){latest=s;const recording=s.state==='RECORDING',finalizing=s.state==='FINALIZING',back=s.back_confirmation||{},previewOnly=s.commissioning_preview_only===true,recordingControls=s.recording_controls_enabled!==false;
 if(previewOnly){document.querySelector('.scope strong').textContent='COMMISSIONING PREVIEW';document.querySelector('.scope span').textContent='Preview only · recording disabled';$('capturehint').textContent='Recording disabled. Verify live camera, points, and audio.'}
 $('statepill').textContent=stateText[s.state]||s.state;$('statepill').style.color=({READY:'var(--green)',RECORDING:'var(--red)',FINALIZING:'var(--amber)',COMPLETE:'var(--green)',PREFLIGHT_FAILED:'var(--red)',ERROR:'var(--red)',ABORTED:'var(--red)'})[s.state]||'var(--muted)';
 $('elapsed').textContent=Number(s.elapsed||0).toFixed(1)+' s';$('framecount').textContent=Number(s.session_recorded_frames||0)+' frames';$('attempt').textContent=(s.idx+1)+' / '+s.n+' · attempt '+s.attempt;$('sessionpath').textContent=s.session_dir||'No session';
 const storage=s.capture_storage;$('storage').textContent=storage?(fmtBytes(storage.usable_bytes_before_safety_reserve)+' · '+fmtRemaining(storage.estimated_remaining_seconds)):'Estimating';
 const proof=s.finalization_summary,tb=proof?.video_timebase;const timebase=tb?.verified?(' · '+Number(tb.encoded_fps).toFixed(2)+' FPS'):'';const audioProof=proof?.audio?.status==='OK'?(' · '+Number(proof.audio.n_samples).toLocaleString()+' audio samples'):'';$('videoresult').textContent=s.state==='COMPLETE'&&proof?.complete?('Verified · '+proof.n_rgb_frames+' video frames'+timebase+audioProof):'Verified on end';
 $('counter').textContent=(s.idx+1)+' / '+s.n;$('tname').textContent=s.point;
 const full=s.target_valid&&s.points_visible===s.n;$('backgate').textContent=full?'10 points locked':back.confirmed?'Back confirmed · locating points':shortText(back.reason||s.tracking_reason||'NO_BACK');$('backgate').className='back'+(full?' ok':'');
 $('pointslabel').textContent=s.points_visible+' / '+s.n;$('pointstat').textContent='Points '+s.points_visible+' / '+s.n;
 $('livefps').textContent='Live FPS '+Number(s.preview_produced_fps||0).toFixed(1);$('sessionframes').textContent='Session frames '+Number(s.session_recorded_frames||0);$('camstat').textContent='Camera '+shortText(s.camera_status)+' · '+fmtAge(s.preview_age_s);const stages=s.perception_stages||{},gateStage=stages.person_gate||{},v9Stage=stages.landmark_model||{};$('modelstat').textContent='Gate '+shortText(gateStage.status||s.model_status)+' · V9 '+shortText(v9Stage.status||'LOADING')+' · '+fmtAge(s.tracking_age_s);const rb=s.robot||{};$('robotstat').textContent=rb.enabled?(rb.ready?'Robot Ready · TX enqueue only · Left can1→can3':'Robot Waiting · Left can1→can3'):'Robot Off';const he=s.handeye||{};$('handeyestat').textContent=he.status==='BOUND_PROVISIONAL'?('Hand-eye · Provisional · '+Number(he.workspace_1sigma_max_mm||0).toFixed(2)+' mm max'):he.status==='BOUND_VALIDATED'?'Hand-eye · Validated':'Hand-eye · Waiting';$('dropstat').textContent='Drops '+(s.model_health?.dropped_before_inference??0)+' / '+(s.capture_ingress?.dropped_before_inference??0);const ct=s.controller||{},armOn=ct.configured===true;$('armctl').hidden=!armOn;if(armOn){const st=ct.state||'idle',dis=ct.disabled===true,iso=ct.energy_isolated===true,cmdOk=ct.disable_command_succeeded!==false,disTxt=iso?'energy-isolated':cmdOk?'motor-disable UNVERIFIED':'DISABLE COMMAND FAILED — arm NOT confirmed disabled';$('armstat').textContent='Arm '+(dis?(cmdOk?'DISABLED · '+disTxt:'⚠ '+disTxt):st==='ready'?('READY · Left can1→can3'+(ct.controller_pid?' · pid '+ct.controller_pid:'')):st==='faulted'?'FAULTED · '+disTxt:st==='idle'?'Idle · not brought up':st);$('armstat').className='armstat '+(st==='ready'&&!dis?'good':(st==='faulted'||dis)?'bad':'warn');$('btnarm').disabled=busy||(st==='ready'&&!dis);$('btnestop').disabled=st==='idle';}
 const au=s.audio||{},signal=au.signal_status||'WAITING',signalPresent=au.signal_present===true;$('audiopill').textContent=signalPresent?'Signal':signal==='SILENT'?'No signal':shortText(signal);$('audiopill').style.color=signalPresent?'var(--green)':au.status==='ERROR'?'var(--red)':'var(--amber)';$('audiodb').textContent=Number(au.peak_db??-120).toFixed(1)+' dBFS';$('audiofill').style.width=Math.max(0,Math.min(100,(Number(au.peak_db??-120)+80)/80*100))+'%';$('audiostatus').textContent=au.commissioned?'Commissioned':signalPresent?'Lung audio detected':signal==='SILENT'?'Check stethoscope':au.enabled?'Waiting for input':'Off';$('btnaudio').disabled=!au.enabled||au.status==='ERROR';
 const fs=s.force_sensor||{},forceN=Number(fs.force_n),forceLimit=Number(fs.hard_limit_force_n),forceStatus=String(fs.status||'WAITING'),forceLive=fs.enabled===true&&Number.isFinite(forceN)&&Number.isFinite(forceLimit)&&forceLimit>0&&['OK','WARN','SCREENED_LIMIT','HARD_LIMIT'].includes(forceStatus),forceLevel=forceLive?Math.max(0,Math.min(100,forceN/forceLimit*100)):0,forceLabel=({OK:'Good',WARN:'Adjust',SCREENED_LIMIT:'High',HARD_LIMIT:'Very high',DISABLED:'Off',STARTING:'Starting',MISSING:'Unavailable',STALE:'Stale',ERROR:'Unavailable',WAITING:'Waiting'})[forceStatus]||shortText(forceStatus),forceColor=fs.status==='WARN'?'var(--amber)':fs.status==='OK'?'var(--green)':forceLive?'var(--red)':'var(--dim)';$('forcepill').textContent=forceLabel;$('forcepill').style.color=forceColor;$('forcefill').style.width=forceLevel+'%';$('forcefill').style.background=forceColor;$('forcebar').setAttribute('aria-valuenow',String(Math.round(forceLevel)));$('forcebar').setAttribute('aria-valuetext',forceLabel);
 $('statusbar').textContent=shortText(s.error||s.fatal_error||s.action_ack||s.tracking_reason||s.state);
 $('live').className='live'+(recording?' on':'');$('recword').textContent=recording?'REC '+Number(s.elapsed).toFixed(1)+'s':finalizing?'Verifying':previewOnly?'Preview only':stateText[s.state]||s.state;
 $('btnstart').disabled=busy||s.start_authorized!==true;$('btnstart').title=s.start_authorized===true?'':(s.start_rejection||'');$('btnstart').textContent=previewOnly?'Recording disabled':(s.state==='COMPLETE'||s.state==='ABORTED'||s.state==='ERROR')?'● New capture':'● Start';$('btnend').disabled=busy||!recordingControls||!recording;
 syncBackButton(back);
 const canConfirm=!busy&&recording&&s.target_valid&&s.points_visible===s.n&&recordingControls;const canRedo=!busy&&recordingControls&&recording;$('btnconfirm').disabled=!canConfirm;$('btnredo').disabled=!canRedo;$('btnprev').disabled=busy||!recordingControls||!recording;$('btnnext').disabled=busy||!recordingControls||!recording;
 pointList(s);blockerList(s.training_blockers||[]);if(s.action_ack&&s.action_ack!==lastActionAck){action(shortText(s.action_ack),s.error?'bad':'good')}if(s.error){action(shortText(s.error),'bad')}lastActionAck=s.action_ack||null}
function schedulePoll(delay){if(pageStopped)return;if(pollTimer)clearTimeout(pollTimer);pollTimer=setTimeout(poll,delay)}
async function poll(){pollTimer=null;if(pageStopped||pollInFlight)return;pollInFlight=true;try{const response=await fetch('/state',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);const state=await response.json();const recovered=transportDown;transportDown=false;render(state);if(recovered&&!state.error&&!state.fatal_error)action(shortText(state.action_ack||state.tracking_reason||state.state),state.action_ack?'good':'')}catch(error){transportDown=true;action('State disconnected: '+error.message,'bad')}finally{pollInFlight=false;schedulePoll(document.hidden?1000:250)}}
async function ctl(name){if(busy)return;busy=true;if(latest)render(latest);try{const payload={action:name};if(name==='confirm_back'){if(!Number.isInteger(displayedPreviewFrameIndex)||typeof displayedPreviewFrameToken!=='string')throw new Error('No displayed camera frame');payload.preview_frame_index=displayedPreviewFrameIndex;payload.preview_frame_token=displayedPreviewFrameToken}const response=await fetch('/control',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();action(shortText(result.ack||result.message||('HTTP '+response.status)),result.ok?'good':'bad')}catch(error){action('Action failed: '+error.message,'bad')}finally{busy=false;await poll()}}
// E-STOP FAST PATH (UI): the disable MUST NOT be gated by `busy` or set it. A doctor has
// to be able to E-stop the arm even while a bring-up/preflight is in flight on THIS same
// page (which holds busy=true). It never touches `busy`, so it can never be blocked by an
// in-flight control action; the backend routes arm_disable straight to disable_controller.
async function estop(){try{const response=await fetch('/control',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'arm_disable'})});const result=await response.json();action(shortText(result.ack||result.message||('HTTP '+response.status)),result.ok?'good':'bad')}catch(error){action('E-stop failed: '+error.message,'bad')}finally{try{await poll()}catch(_e){}}}
async function sha256Hex(buffer){const digest=await crypto.subtle.digest('SHA-256',buffer);return Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,'0')).join('')}
function afterPaint(){return new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))}
function scheduleCamera(delay){if(pageStopped)return;if(cameraTimer)clearTimeout(cameraTimer);cameraTimer=setTimeout(refreshCamera,delay)}
async function refreshCamera(){cameraTimer=null;if(pageStopped||cameraInFlight)return;if(document.hidden){scheduleCamera(1000);return}cameraInFlight=true;let bitmap=null;try{const query=new URLSearchParams({after:String(lastRenderedFrameIndex??-1)});if(Number.isInteger(lastRenderedGeneration))query.set('generation',String(lastRenderedGeneration));const response=await fetch('/frame.jpg?'+query,{cache:'no-store'});if(response.status===204)return;if(!response.ok)throw new Error('video HTTP '+response.status);const frameIndex=Number(response.headers.get('X-Preview-Frame-Index')),frameGeneration=Number(response.headers.get('X-Preview-Generation')),frameToken=response.headers.get('X-Preview-Frame-Token'),confirmable=response.headers.get('X-Preview-Confirmable')==='1';if(!Number.isInteger(frameIndex)||frameIndex<0||!Number.isInteger(frameGeneration)||frameGeneration<0||(confirmable&&!/^[0-9a-f]{64}\.[0-9a-f]{64}$/.test(frameToken||''))||(!confirmable&&frameToken!==null))throw new Error('video frame identity invalid');const bytes=await response.arrayBuffer();if(confirmable){const expectedDigest=frameToken.slice(0,64);if(await sha256Hex(bytes)!==expectedDigest)throw new Error('video frame digest mismatch')}const blob=new Blob([bytes],{type:'image/jpeg'});bitmap=await createImageBitmap(blob);if(!bitmap.width||!bitmap.height)throw new Error('video decode failed');const canvas=$('camera');if(canvas.width!==bitmap.width)canvas.width=bitmap.width;if(canvas.height!==bitmap.height)canvas.height=bitmap.height;const context=canvas.getContext('2d',{alpha:false});if(!context)throw new Error('video canvas unavailable');displayedPreviewFrameIndex=null;displayedPreviewFrameToken=null;syncBackButton();context.drawImage(bitmap,0,0);lastRenderedFrameIndex=frameIndex;lastRenderedGeneration=frameGeneration;bitmap.close();bitmap=null;await afterPaint();displayedPreviewFrameIndex=confirmable?frameIndex:null;displayedPreviewFrameToken=confirmable?frameToken:null;syncBackButton()}catch(error){if(!pageStopped)action('Video interrupted · reconnecting…','bad')}finally{if(bitmap)bitmap.close();cameraInFlight=false;scheduleCamera(document.hidden?1000:40)}}
function stopAudio(){listening=false;if(audioTimer)clearTimeout(audioTimer);audioTimer=null;audioCursor=null;audioNextAt=0;$('btnaudio').textContent='▶ Monitor';$('btnaudio').className='listen'}
async function pumpAudio(){if(!listening)return;try{const query=new URLSearchParams({samples:'4410'});if(audioCursor!==null)query.set('cursor',String(audioCursor));const response=await fetch('/audio/pcm?'+query,{cache:'no-store'});if(!response.ok)throw new Error('audio HTTP '+response.status);const audioStatus=response.headers.get('X-Audio-Status')||'NO_DATA';if(audioStatus!=='OK')throw new Error('Audio input '+audioStatus.toLowerCase());const rate=Number(response.headers.get('X-Audio-Sample-Rate'));const overrun=response.headers.get('X-Audio-Overrun')==='1';audioCursor=Number(response.headers.get('X-Audio-Next-Sample'));const raw=await response.arrayBuffer();if(raw.byteLength&&Number.isFinite(rate)&&rate>0){const pcm=new Int16Array(raw),buffer=audioContext.createBuffer(1,pcm.length,rate),channel=buffer.getChannelData(0);for(let i=0;i<pcm.length;i++)channel[i]=pcm[i]/32768;const lead=.08;if(overrun||audioNextAt<audioContext.currentTime||audioNextAt>audioContext.currentTime+.6)audioNextAt=audioContext.currentTime+lead;const source=audioContext.createBufferSource();source.buffer=buffer;source.connect(audioContext.destination);source.onended=()=>source.disconnect();source.start(audioNextAt);audioNextAt+=buffer.duration}audioTimer=setTimeout(pumpAudio,60)}catch(error){action('Audio monitoring stopped: '+error.message,'bad');stopAudio()}}
async function toggleAudio(){if(listening){stopAudio();return}try{audioContext=audioContext||new AudioContext();await audioContext.resume();listening=true;audioCursor=null;audioNextAt=0;$('btnaudio').textContent='■ Stop audio';$('btnaudio').className='listen on';pumpAudio()}catch(error){action('Browser audio unavailable: '+error.message,'bad');stopAudio()}}
$('btnback').onclick=()=>ctl('confirm_back');$('btnstart').onclick=()=>ctl('start');$('btnend').onclick=()=>ctl('end');$('btnconfirm').onclick=()=>ctl('confirm');$('btnredo').onclick=()=>ctl('redo');$('btnprev').onclick=()=>ctl('prev');$('btnnext').onclick=()=>ctl('next');
$('btnarm').onclick=()=>{if(confirm('Bring up the LEFT arm controller? The arm will energize and move.'))ctl('arm_bringup')};$('btnestop').onclick=()=>estop();
$('btnaudio').onclick=toggleAudio;
addEventListener('keydown',event=>{if(event.repeat||event.metaKey||event.ctrlKey||event.altKey)return;const key=event.key;if((key==='b'||key==='B')&&!$('btnback').disabled)ctl('confirm_back');else if(key===' '){event.preventDefault();if(!$('btnconfirm').disabled)ctl('confirm')}else if((key==='r'||key==='R')&&!$('btnredo').disabled)ctl('redo');else if(key==='ArrowLeft'&&!$('btnprev').disabled)ctl('prev');else if(key==='ArrowRight'&&!$('btnnext').disabled)ctl('next');else if(key==='s'||key==='S'){if(latest?.state==='RECORDING')ctl('end');else if(!$('btnstart').disabled)ctl('start')}else if(key==='f'||key==='F')$('feed').requestFullscreen?.()});
addEventListener('visibilitychange',()=>{if(document.hidden){if(listening)stopAudio();return}schedulePoll(0);scheduleCamera(0)});
addEventListener('pagehide',()=>{pageStopped=true;if(pollTimer)clearTimeout(pollTimer);if(cameraTimer)clearTimeout(cameraTimer);pollTimer=null;cameraTimer=null;stopAudio()});
addEventListener('pageshow',()=>{if(!pageStopped)return;pageStopped=false;schedulePoll(0);scheduleCamera(0)});
poll();refreshCamera();
</script></body></html>""".encode()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="synthetic camera + simulated sensors (no hardware)")
    ap.add_argument(
        "--commissioning-preview-only",
        action="store_true",
        help=(
            "run real camera, perception, and audio for commissioning while "
            "server-side recording controls remain disabled"
        ),
    )
    ap.add_argument(
        "--production-teleop",
        action="store_true",
        help=(
            "record promotable production teleoperation sessions "
            "(ProductionTeleopCapture + production_teleop_v4 contract); requires a "
            "production activation and a live left OpenArm"
        ),
    )
    ap.add_argument(
        "--commissioning-hall-monitor-only",
        action="store_true",
        help=(
            "show calibrated Hall force in preview without claiming audio-force "
            "synchronization or training eligibility"
        ),
    )
    ap.add_argument("--commissioning-handeye-release", default=None)
    ap.add_argument(
        "--commissioning-handeye-artifact-sha256", default=None,
    )
    ap.add_argument(
        "--commissioning-handeye-complete-sha256", default=None,
    )
    ap.add_argument("--camera-backend", default="orbbec", help="orbbec (linbox) | avfoundation (webcam) | fake")
    ap.add_argument("--camera-index", type=int, default=0, help="UVC device index (Mac built-in cam is often 1)")
    ap.add_argument("--camera-id", default=None, help="required stable Orbbec serial in real mode")
    ap.add_argument("--camera-width", type=int, default=None)
    ap.add_argument("--camera-height", type=int, default=None)
    ap.add_argument("--camera-fps", type=int, default=None)
    ap.add_argument("--camera-color-format", default=None)
    ap.add_argument("--depth-width", type=int, default=None)
    ap.add_argument("--depth-height", type=int, default=None)
    ap.add_argument("--depth-fps", type=int, default=None)
    ap.add_argument("--depth-format", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-sha256", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--base-model-sha256", default=None)
    ap.add_argument("--model-release-id", default=None)
    ap.add_argument("--posterior-gate-checkpoint", default=None)
    ap.add_argument("--posterior-gate-checkpoint-sha256", default=None)
    ap.add_argument("--posterior-gate-release-id", default=None)
    ap.add_argument("--runtime-release-id", default=None)
    ap.add_argument("--runtime-lock", default=None)
    ap.add_argument("--runtime-lock-sha256", default=None)
    ap.add_argument("--code-release-id", default=None)
    ap.add_argument("--ffmpeg-path", default=None)
    ap.add_argument("--ffmpeg-sha256", default=None)
    ap.add_argument("--diagnostic-fps", type=float, default=20.0)
    ap.add_argument("--diagnostic-min-free-gib", type=float, default=5.0)
    ap.add_argument("--planned-session-duration-s", type=float, default=60.0)
    ap.add_argument("--capture-rate-margin", type=float, default=1.3)
    ap.add_argument("--depth-compression-ratio", type=float, default=1.0,
                    help="storage-gate depth compression assumption; default 1.0 = raw worst-case (noisy depth can be ~1x). Raise only after validating the scene.")
    ap.add_argument("--non-depth-capture-bytes-per-s", type=float, default=12_000_000,
                    help="conservative RGB-video + audio + force + tracker byte rate for the storage gate")
    ap.add_argument("--teleop-launch-cmd", default=None,
                    help="shell command that launches the pinned left teleop controller "
                         "(can1 leader -> can3 follower); enables the console Bring-up-arm button")
    ap.add_argument("--controller-disable-cmd", default=None,
                    help="shell command that energy-isolates the left arm (direct-disable)")
    ap.add_argument("--controller-preflight-cmd", default=None,
                    help="optional shell command run synchronously BEFORE the controller "
                         "is launched (clear_faults + telemetry-socket dir); lets the "
                         "controller binary be launched directly for a sound exe attestation")
    ap.add_argument("--controller-readiness-timeout-s", type=float, default=15.0)
    ap.add_argument("--max-tracking-age-ms", type=float, default=250.0)
    ap.add_argument("--capture-queue-frames", type=int, default=8)
    ap.add_argument(
        "--camera-delivery-frames", type=int, default=32,
        help="bounded non-overwrite CameraStream FIFO capacity",
    )
    ap.add_argument("--camera-reopen-interval-s", type=float, default=2.0)
    ap.add_argument(
        "--audio-device", default=None,
        help="exact PortAudio input identity owned by Linbox",
    )
    ap.add_argument("--audio-sample-rate", type=int, default=44_100)
    ap.add_argument("--audio-channels", type=int, default=2)
    ap.add_argument("--audio-pick-channel", type=int, default=None)
    ap.add_argument("--audio-expected-usb-vid", type=lambda value: int(value, 0), default=None)
    ap.add_argument("--audio-expected-usb-pid", type=lambda value: int(value, 0), default=None)
    ap.add_argument("--audio-expected-usb-serial", default=None)
    ap.add_argument("--audio-expected-usb-port-path", default=None)
    ap.add_argument(
        "--require-audio", action="store_true",
        help="fail closed unless the commissioned stethoscope stream is live",
    )
    ap.add_argument("--hall-calibration", default=None)
    ap.add_argument("--hall-calibration-sha256", default=None)
    ap.add_argument("--hall-port", default=None)
    ap.add_argument("--hall-expected-usb-vid", type=lambda value: int(value, 0), default=None)
    ap.add_argument("--hall-expected-usb-pid", type=lambda value: int(value, 0), default=None)
    ap.add_argument("--hall-expected-usb-serial", default=None)
    ap.add_argument("--sync-calibration", default=None)
    ap.add_argument("--sync-calibration-sha256", default=None)
    ap.add_argument(
        "--require-force", action="store_true",
        help="require the calibrated Hall stream and physical audio-force sync",
    )
    ap.add_argument(
        "--production-blocker", action="append", default=None,
        help="repeatable ADDITIONAL commissioning blocker, appended to the built-in "
             "fail-closed list. It can only ADD blockers, never clear them: the "
             "commissioning-complete set is cleared solely by SHA-bound activation "
             "evidence, never by an unauthenticated CLI string.",
    )
    ap.add_argument(
        "--production-activation", default=None,
        help="absolute immutable production activation.json (real mode)",
    )
    ap.add_argument(
        "--production-activation-sha256", default=None,
        help="independent SHA-256 pin for production activation.json",
    )
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    return ap


def _production_capture_mode(args):
    """Return the exact mode a hardware timing artifact must authorize."""

    if args.camera_backend != "orbbec":
        raise ValueError("production activation requires the Orbbec backend")
    fields = {
        "camera_id": args.camera_id,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "camera_fps": args.camera_fps,
        "camera_color_format": args.camera_color_format,
        "depth_width": args.depth_width,
        "depth_height": args.depth_height,
        "depth_fps": args.depth_fps,
        "depth_format": args.depth_format,
    }
    missing = [name for name, value in fields.items() if value is None or value == ""]
    if missing:
        raise ValueError(
            "production activation requires pinned camera mode fields: "
            + ", ".join(missing)
        )
    return {
        "align_mode": "software_depth_to_color",
        "backend": "orbbec",
        "camera_id": args.camera_id,
        "color_format": args.camera_color_format,
        "depth_mode": {
            "format": args.depth_format,
            "fps": float(args.depth_fps),
            "height": int(args.depth_height),
            "width": int(args.depth_width),
        },
        "fps": float(args.camera_fps),
        "frame_aggregate_mode": "full_frame_require",
        "height": int(args.camera_height),
        "time_source": "orbbec_device_and_host_arrival",
        "width": int(args.camera_width),
    }


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.commissioning_preview_only and a.fake:
        parser.error(
            "--commissioning-preview-only requires real hardware; "
            "--fake is forbidden"
        )
    handeye_fields = {
        "--commissioning-handeye-release": a.commissioning_handeye_release,
        "--commissioning-handeye-artifact-sha256": (
            a.commissioning_handeye_artifact_sha256
        ),
        "--commissioning-handeye-complete-sha256": (
            a.commissioning_handeye_complete_sha256
        ),
    }
    supplied_handeye = [
        name for name, value in handeye_fields.items()
        if value not in {None, ""}
    ]
    if supplied_handeye and not a.commissioning_preview_only:
        parser.error("commissioning hand-eye release is preview-only")
    if supplied_handeye and len(supplied_handeye) != len(handeye_fields):
        parser.error(
            "commissioning hand-eye release requires path, artifact SHA-256, "
            "and COMPLETE SHA-256"
        )
    if a.commissioning_hall_monitor_only and not a.commissioning_preview_only:
        parser.error("--commissioning-hall-monitor-only requires preview mode")
    preview_hall_fields = {
        "--hall-calibration": a.hall_calibration,
        "--hall-calibration-sha256": a.hall_calibration_sha256,
        "--hall-port": a.hall_port,
        "--hall-expected-usb-vid": a.hall_expected_usb_vid,
        "--hall-expected-usb-pid": a.hall_expected_usb_pid,
        "--hall-expected-usb-serial": a.hall_expected_usb_serial,
    }
    supplied_preview_hall = [
        name for name, value in preview_hall_fields.items()
        if value not in {None, ""}
    ]
    if a.commissioning_preview_only:
        if a.commissioning_hall_monitor_only:
            missing_hall = [
                name for name, value in preview_hall_fields.items()
                if value in {None, ""}
            ]
            if missing_hall:
                parser.error(
                    "commissioning Hall monitor requires: "
                    + ", ".join(missing_hall)
                )
            if a.sync_calibration or a.sync_calibration_sha256 or a.require_force:
                parser.error(
                    "unsynchronized Hall monitor forbids sync inputs and "
                    "--require-force"
                )
        elif (
            supplied_preview_hall
            or a.sync_calibration
            or a.sync_calibration_sha256
            or a.require_force
        ):
            parser.error(
                "preview Hall inputs require --commissioning-hall-monitor-only"
            )
    if not a.fake and (not a.ffmpeg_path or not a.ffmpeg_sha256):
        parser.error("real collection requires --ffmpeg-path and --ffmpeg-sha256")
    if not a.fake and not a.commissioning_preview_only and (
        not a.production_activation or not a.production_activation_sha256
    ):
        parser.error(
            "real collection requires --production-activation and "
            "--production-activation-sha256"
        )
    if a.production_teleop and (
        a.fake
        or a.commissioning_preview_only
        or not a.production_activation
        or not a.production_activation_sha256
    ):
        parser.error(
            "--production-teleop requires a real production activation and is "
            "incompatible with --fake / --commissioning-preview-only"
        )
    if not a.fake and not a.commissioning_preview_only:
        force_fields = {
            "--hall-calibration": a.hall_calibration,
            "--hall-calibration-sha256": a.hall_calibration_sha256,
            "--hall-port": a.hall_port,
            "--hall-expected-usb-vid": a.hall_expected_usb_vid,
            "--hall-expected-usb-pid": a.hall_expected_usb_pid,
            "--hall-expected-usb-serial": a.hall_expected_usb_serial,
            "--sync-calibration": a.sync_calibration,
            "--sync-calibration-sha256": a.sync_calibration_sha256,
        }
        missing_force = [
            name for name, value in force_fields.items()
            if value is None or value == ""
        ]
        if missing_force:
            parser.error(
                "real collection requires immutable Hall/sync inputs: "
                + ", ".join(missing_force)
            )
        if not a.require_audio or a.audio_pick_channel is None:
            parser.error(
                "real Hall-force collection requires --require-audio and "
                "--audio-pick-channel"
            )
    production_blockers = _resolve_production_blockers(a.production_blocker)
    targets = load_targets(a.targets)
    production_runtime = None
    commissioning_handeye = None
    if a.commissioning_preview_only and len(supplied_handeye) == len(handeye_fields):
        try:
            commissioning_handeye = load_completed_handeye_release(
                a.commissioning_handeye_release,
                expected_artifact_sha256=(
                    a.commissioning_handeye_artifact_sha256
                ),
                expected_complete_sha256=(
                    a.commissioning_handeye_complete_sha256
                ),
                expected_camera_id=a.camera_id,
            )
        except (ProductionRuntimeActivationError, TypeError, ValueError) as exc:
            parser.error(f"commissioning hand-eye release rejected: {exc}")
    if not a.fake and not a.commissioning_preview_only:
        try:
            production_runtime = load_production_runtime_assembly(
                a.production_activation,
                expected_activation_sha256=a.production_activation_sha256,
                point_names=tuple(target.name for target in targets),
                expected_camera_id=a.camera_id,
                expected_capture_mode=_production_capture_mode(a),
            )
        except (ProductionRuntimeActivationError, TypeError, ValueError) as exc:
            parser.error(f"production activation rejected: {exc}")

    session_root = a.out or os.path.join(ROOT, "data", "sessions")
    reconciled = reconcile_orphan_diagnostic_sessions(
        session_root,
        video_recoverer=lambda path: recover_orphan_diagnostic_video(
            path, ffmpeg_path=a.ffmpeg_path, ffmpeg_sha256=a.ffmpeg_sha256,
        ),
    )
    for path in reconciled:
        print(f"[collect] marked interrupted diagnostic session ABORTED: {path}")
    # Console-owned arm controller lifecycle (both commands required together so a
    # bring-up can always be energy-isolated).
    controller_launcher = None
    controller_disable = None
    controller_preflight = None
    if a.teleop_launch_cmd or a.controller_disable_cmd:
        if not (a.teleop_launch_cmd and a.controller_disable_cmd):
            parser.error("--teleop-launch-cmd and --controller-disable-cmd must be given together")
        from ausc.perception.robot.controller_launch import (
            make_subprocess_direct_disable,
            make_subprocess_launcher,
            make_subprocess_preflight,
            parse_command,
        )
        controller_launcher = make_subprocess_launcher(
            parse_command(a.teleop_launch_cmd, label="--teleop-launch-cmd"),
        )
        controller_disable = make_subprocess_direct_disable(
            parse_command(a.controller_disable_cmd, label="--controller-disable-cmd"),
        )
        if a.controller_preflight_cmd:
            controller_preflight = make_subprocess_preflight(
                parse_command(
                    a.controller_preflight_cmd, label="--controller-preflight-cmd"
                ),
            )
    col = None
    server = None
    try:
        col = Collector(
            targets,
            fake=a.fake,
            camera_backend=a.camera_backend,
            camera_index=a.camera_index,
            camera_id=a.camera_id,
            camera_width=a.camera_width,
            camera_height=a.camera_height,
            camera_fps=a.camera_fps,
            camera_color_format=a.camera_color_format,
            depth_width=a.depth_width,
            depth_height=a.depth_height,
            depth_fps=a.depth_fps,
            depth_format=a.depth_format,
            out_dir=session_root,
            checkpoint=a.checkpoint,
            checkpoint_sha256=a.checkpoint_sha256,
            base_model=a.base_model,
            base_model_sha256=a.base_model_sha256,
            model_release_id=a.model_release_id,
            posterior_gate_checkpoint=a.posterior_gate_checkpoint,
            posterior_gate_checkpoint_sha256=a.posterior_gate_checkpoint_sha256,
            posterior_gate_release_id=a.posterior_gate_release_id,
            runtime_release_id=a.runtime_release_id,
            runtime_lock=a.runtime_lock,
            runtime_lock_sha256=a.runtime_lock_sha256,
            code_release_id=a.code_release_id,
            ffmpeg_path=a.ffmpeg_path,
            ffmpeg_sha256=a.ffmpeg_sha256,
            diagnostic_fps=a.diagnostic_fps,
            diagnostic_min_free_bytes=int(a.diagnostic_min_free_gib * 1024 ** 3),
            planned_session_duration_s=a.planned_session_duration_s,
            capture_rate_margin=a.capture_rate_margin,
            depth_compression_ratio=a.depth_compression_ratio,
            non_depth_capture_bytes_per_s=a.non_depth_capture_bytes_per_s,
            max_tracking_age_s=a.max_tracking_age_ms / 1000.0,
            capture_queue_frames=a.capture_queue_frames,
            camera_delivery_frames=a.camera_delivery_frames,
            camera_reopen_interval_s=a.camera_reopen_interval_s,
            audio_device=a.audio_device,
            audio_sample_rate=a.audio_sample_rate,
            audio_channels=a.audio_channels,
            audio_pick_channel=a.audio_pick_channel,
            audio_expected_usb_vid=a.audio_expected_usb_vid,
            audio_expected_usb_pid=a.audio_expected_usb_pid,
            audio_expected_usb_serial=a.audio_expected_usb_serial,
            audio_expected_usb_port_path=a.audio_expected_usb_port_path,
            require_audio=a.require_audio,
            hall_calibration=a.hall_calibration,
            hall_calibration_sha256=a.hall_calibration_sha256,
            hall_port=a.hall_port,
            hall_expected_usb_vid=a.hall_expected_usb_vid,
            hall_expected_usb_pid=a.hall_expected_usb_pid,
            hall_expected_usb_serial=a.hall_expected_usb_serial,
            sync_calibration=a.sync_calibration,
            sync_calibration_sha256=a.sync_calibration_sha256,
            require_force=(
                False
                if a.commissioning_preview_only
                else (not a.fake or a.require_force)
            ),
            require_robot=not a.fake and not a.commissioning_preview_only,
            robot_capture_coordinator=(
                None
                if production_runtime is None
                else production_runtime.robot_capture_coordinator
            ),
            controller_launcher=controller_launcher,
            controller_disable=controller_disable,
            controller_preflight=controller_preflight,
            # Independent motor-disable readback: injected from the production runtime when
            # it provides one. It stays None until the STATE frame is decoded into per-motor
            # torque-off flags (a go-live telemetry item); None keeps `energy_isolated`
            # honestly UNVERIFIED and, with a verifier present-and-False, keeps the arm
            # unlatched so the watchdog retries and no replacement is allowed.
            controller_verify_motor_disable=(
                None
                if production_runtime is None
                else getattr(production_runtime, "motor_disable_verifier", None)
            ),
            controller_readiness_timeout_s=a.controller_readiness_timeout_s,
            robot_mask_provider=(
                None
                if production_runtime is None
                else production_runtime.robot_mask_provider
            ),
            handeye_activation=(
                commissioning_handeye
                if production_runtime is None
                else production_runtime.handeye
            ),
            back_mask_provider=(
                None
                if production_runtime is None
                else production_runtime.back_mask_provider
            ),
            patient_pose_provider=(
                None
                if production_runtime is None
                else production_runtime.patient_pose_provider
            ),
            occlusion_runtime=(
                None
                if production_runtime is None
                else production_runtime.occlusion_runtime
            ),
            production_blockers=production_blockers,
            commissioning_preview_only=a.commissioning_preview_only,
            production_teleop=a.production_teleop,
            commissioning_hall_monitor_only=(
                a.commissioning_hall_monitor_only
            ),
        )
        src = "FAKE" if a.fake else f"{a.camera_backend}[{a.camera_index}]"
        print(f"[collect] console at http://localhost:{a.port}  ({src})  keys: SPACE=confirm R=redo ←→=points S=rec")
        print(f"[collect] remote browser: on your laptop run  ssh -L {a.port}:localhost:{a.port} <this-host>  "
              f"then open http://localhost:{a.port}")
        server = CollectionHTTPServer((a.bind, a.port), make_handler(col, PAGE))

        def stop_server(_signum, _frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop_server)
        signal.signal(signal.SIGINT, stop_server)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        if col is not None:
            col.shutdown()
        if production_runtime is not None:
            production_runtime.close()


if __name__ == "__main__":
    main()
