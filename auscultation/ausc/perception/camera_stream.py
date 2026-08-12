"""Continuous, non-blocking camera reader for the auscultation pipeline.

Third acquisition primitive alongside `ausc/audio/audio_stream.AudioStream` and
`mechanical_sensor/force_feedback/force_sensor.ForceSensor`. Same contract: a background
thread grabs frames into a one-slot buffer; any loop calls the **non-blocking**
`latest_frame()` and gets the most recent frame plus its `time.monotonic()` timestamp, so
camera / audio / force all co-register on one host clock (see
`docs/realtime_multimodal_pipeline.md` §1).

Two backends:
  - **"orbbec"** (default): Orbbec SDK (`pyorbbecsdk`) -> **RGB + DEPTH** from a Femto Bolt.
    Use this for the depth camera. RGB and depth come from one frameset, optionally aligned.
  - **"avfoundation"**: plain OpenCV/UVC capture -> **RGB only**, for a generic webcam.

macOS note: do NOT open the Femto Bolt with the avfoundation backend — macOS's camera
subsystem then claims the UVC device and the Orbbec SDK can no longer open it (a replug is
needed to recover). Use the "orbbec" backend exclusively for the depth camera.

    with CameraStream(backend="orbbec") as cam:
        f = cam.latest_frame()                 # f.frame = RGB HxWx3, f.depth = uint16 HxW (mm via depth_scale)
        if f.status == "OK":
            obs["observation.images.cam"] = f.frame
            obs["observation.depth.cam"]  = f.depth
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

STATUS_OK = "OK"
STATUS_STALE = "STALE"
STATUS_NO_DATA = "NO_DATA"
DEFAULT_DELIVERY_CAPACITY_FRAMES = 32


def _orbbec_device_time_ms(frame) -> float | None:  # noqa: ANN001
    """Return the SDK's capture/device timestamp without inventing one when absent."""
    if frame is None or not hasattr(frame, "get_timestamp"):
        return None
    try:
        value = float(frame.get_timestamp())
    except (TypeError, ValueError, RuntimeError):
        return None
    return value if np.isfinite(value) else None


def _as_orbbec_frameset(frame):  # noqa: ANN001
    """Normalize v1/v2 AlignFilter output to an object exposing color/depth frames."""
    if frame is None:
        return None
    if hasattr(frame, "get_color_frame") and hasattr(frame, "get_depth_frame"):
        return frame
    if hasattr(frame, "as_frame_set"):
        candidate = frame.as_frame_set()
        if (candidate is not None and hasattr(candidate, "get_color_frame")
                and hasattr(candidate, "get_depth_frame")):
            return candidate
    return None


def _sdk_format_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(value).rsplit(".", 1)[-1]


def _active_orbbec_color_distortion(active_video_profile) -> dict[str, object]:  # noqa: ANN001
    """Copy the exact enabled profile coefficients without inventing a model."""

    try:
        active_distortion = active_video_profile.get_distortion()
        distortion_by_name = {
            name: float(getattr(active_distortion, name))
            for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
        }
    except Exception as exc:  # noqa: BLE001 - SDK exception type varies by release
        raise RuntimeError("active Orbbec color distortion is unavailable or invalid") from exc
    if not all(np.isfinite(value) for value in distortion_by_name.values()):
        raise RuntimeError("active Orbbec color distortion contains non-finite coefficients")
    # pyorbbecsdk2 2.1.1 exposes these coefficients on VideoStreamProfile but
    # does not expose the associated OBCameraDistortionModel. Keep native field
    # order and make that uncertainty explicit; consumers must not silently feed
    # these values into an OpenCV Brown/Inverse/Modified-Brown implementation.
    coefficient_order = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    return {
        "model": "model_api_unexposed",
        "coefficient_order": list(coefficient_order),
        "coefficients": [distortion_by_name[name] for name in coefficient_order],
    }


@dataclass(frozen=True)
class CameraFrame:
    """One immutable frame. `frame` (RGB) / `depth` (uint16) are None until the first grab."""

    frame: Optional[np.ndarray]   # HxWx3 uint8 RGB, or None
    t: float                      # host monotonic time of the grab (s)
    index: int                    # frame counter since start
    status: str                   # OK | STALE | NO_DATA
    depth: Optional[np.ndarray] = None   # HxW uint16 (multiply by depth_scale for mm)
    depth_scale: float = 1.0             # mm per depth unit
    age_s: float = 0.0
    device_t_ms: Optional[float] = None  # camera clock at exposure/capture, if backend provides it
    time_source: str = "host_arrival"


@dataclass(frozen=True)
class CameraEndBoundary:
    """Atomically latched camera state at the formal recording End boundary."""

    t_end: float
    frame: Optional[CameraFrame]
    proof: dict[str, object]


@dataclass(frozen=True)
class CameraStartBoundary:
    """Atomically armed lossless-delivery interval after recorder preparation."""

    interval_id: int
    t_start: float
    frame: Optional[CameraFrame]
    proof: dict[str, object]


def _ob_color_to_rgb(color_frame, ob) -> Optional[np.ndarray]:
    """Decode an Orbbec color frame into storage owned beyond the SDK frame lifetime."""
    import cv2

    w, h = color_frame.get_width(), color_frame.get_height()
    buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    F = ob.OBFormat
    fmt = color_frame.get_format()

    def is_(name: str) -> bool:  # tolerate SDK builds that lack a given format constant
        return hasattr(F, name) and fmt == getattr(F, name)

    try:
        if is_("RGB"):
            return np.ascontiguousarray(buf.reshape(h, w, 3)).copy()
        if is_("BGR"):
            return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
        if is_("MJPG"):
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if is_("YUYV"):
            return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
        if is_("UYVY"):
            return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2RGB_UYVY)
        if is_("NV12"):
            return cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2RGB_NV12)
    except Exception:  # noqa: BLE001 - bad frame -> skip
        return None
    return None  # unknown format


def _ob_depth_to_array(depth_frame) -> np.ndarray:  # noqa: ANN001
    """Detach one uint16 depth plane from the SDK-owned frame buffer."""

    depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        depth_frame.get_height(), depth_frame.get_width(),
    )
    return np.ascontiguousarray(depth).copy()


class CameraStream:
    """Background camera capture (Orbbec RGB+depth, or OpenCV RGB). Open with `start()`
    (or `with`), then call the non-blocking `latest_frame()` from any loop."""

    def __init__(
        self,
        backend: str = "orbbec",
        index: int = 0,
        camera_id: str | None = None,
        align_depth_to_color: bool = True,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        color_format: str | None = None,
        depth_mode: dict[str, object] | None = None,
        rgb: bool = True,
        stale_timeout_ms: float = 500.0,
        delivery_capacity_frames: int = DEFAULT_DELIVERY_CAPACITY_FRAMES,
        recording_frame_divisor: int = 1,
    ) -> None:
        self.backend = backend
        self.index = int(index)
        self.camera_id = camera_id.strip() if isinstance(camera_id, str) else None
        self.align = align_depth_to_color
        self.width = width
        self.height = height
        self.fps = fps
        if self.fps is not None:
            try:
                requested_fps = float(self.fps)
            except (TypeError, ValueError) as exc:
                raise ValueError("camera fps must be finite and positive") from exc
            if not np.isfinite(requested_fps) or requested_fps <= 0:
                raise ValueError("camera fps must be finite and positive")
        self.color_format = color_format.strip() if isinstance(color_format, str) else None
        self.depth_mode = dict(depth_mode) if depth_mode is not None else None
        self.rgb = rgb
        self.stale_timeout_s = stale_timeout_ms / 1000.0
        self.delivery_capacity_frames = int(delivery_capacity_frames)
        # Recording-set subsampling (2026-08-12, owner-authorized 15 fps
        # recording): only every Nth captured frame enters the lossless
        # delivery FIFO, phase-anchored per interval at the Start boundary
        # (start_boundary_index + 1) so the first delivered frame is always
        # boundary+1 regardless of boundary parity. Default 1 keeps every
        # other consumer of this class (record_session.py strict-contiguous
        # policy, all existing tests) byte-identical.
        if type(recording_frame_divisor) is not int or recording_frame_divisor < 1:
            raise ValueError("recording frame divisor must be an integer >= 1")
        self.recording_frame_divisor = recording_frame_divisor
        self._delivery_phase_anchor = 1
        if self.delivery_capacity_frames <= 0:
            raise ValueError("camera delivery FIFO capacity must be positive")
        self._cap = None          # cv2 VideoCapture (avfoundation)
        self._pipe = None         # pyorbbecsdk Pipeline (orbbec)
        self._context = None
        self._device = None
        self._ob = None
        self._align = None
        self._thread: threading.Thread | None = None
        self._thread_error: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest: CameraFrame | None = None
        self._count = 0
        # A frame's host timestamp and its in-flight registration are written
        # under the same condition used to sample End.  This closes the gap in
        # which End could otherwise miss an already-arrived Orbbec frameset
        # that was still being aligned or decoded.
        self._next_inflight_id = 0
        self._inflight_id: int | None = None
        self._inflight_host_t: float | None = None
        self._end_boundary_t: float | None = None
        self._end_boundary_snapshot: CameraFrame | None = None
        self._end_boundary_target_id: int | None = None
        self._end_boundary_target_completed = False
        self._end_boundary_target_produced_frame: bool | None = None
        self._end_boundary_result: CameraEndBoundary | None = None
        # Preview/inference read `_latest`; lossless recording consumes this
        # separate explicitly armed FIFO. Never use deque(maxlen): a full queue
        # must be visible and fail closed instead of overwriting anatomy frames.
        self._delivery_queue: deque[CameraFrame] = deque()
        self._delivery_interval_generation = 0
        self._delivery_interval_id: int | None = None
        self._delivery_interval_active = False
        self._delivery_accepting = False
        self._delivery_closing = False
        self._delivery_start_t: float | None = None
        self._delivery_start_boundary_snapshot: CameraFrame | None = None
        self._delivery_start_target_id: int | None = None
        self._delivery_start_target_completed = False
        self._delivery_start_target_produced_frame: bool | None = None
        self._delivery_enqueued = 0
        self._delivery_delivered = 0
        self._delivery_overflow_total = 0
        self._delivery_interval_overflow = 0
        self._delivery_rejected_after_overflow = 0
        self._delivery_first_enqueued_index: int | None = None
        self._delivery_last_enqueued_index: int | None = None
        self._delivery_last_delivered_index: int | None = None
        self._delivery_first_overflow_index: int | None = None
        self._delivery_max_depth = 0
        self._delivery_discarded = 0
        self.negotiated_mode: dict[str, object] | None = None
        self.device_identity: str | None = None
        self.color_intrinsics: dict[str, float | int] | None = None
        self.color_distortion: dict[str, object] | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "CameraStream":
        try:
            if self.backend == "orbbec":
                self._start_orbbec(); target = self._run_orbbec
            elif self.backend == "fake":        # synthetic RGB+depth (no hardware; for tests/CI)
                self.negotiated_mode = {
                    "width": int(self.width or 640),
                    "height": int(self.height or 480),
                    "fps": float(self.fps or 30.0),
                    "color_format": "synthetic_rgb",
                    "depth_mode": {
                        "width": int(self.width or 640), "height": int(self.height or 480),
                        "fps": float(self.fps or 30.0), "format": "synthetic_u16",
                    },
                    "align_mode": "disabled",
                    "frame_aggregate_mode": "not_applicable",
                }
                target = self._run_fake
            else:
                self._start_avfoundation(); target = self._run_cv2
            self._stop.clear()
            self._thread_error = None
            self._thread = threading.Thread(
                target=self._run_guarded,
                args=(target,),
                name="camera-stream",
                daemon=True,
            )
            self._thread.start()
            return self
        except BaseException:
            # A profile getter or Thread.start can fail after the SDK has claimed
            # the USB device. Always release partial backend state before re-raising.
            try:
                self.stop()
            except BaseException:
                pass
            raise

    def _run_guarded(self, target) -> None:  # noqa: ANN001
        """Turn an uncaught backend-loop exception into visible health evidence."""

        try:
            target()
        except BaseException as exc:  # thread death must never be silent
            with self._condition:
                self._thread_error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                self._condition.notify_all()

    def _start_orbbec(self) -> None:
        import pyorbbecsdk as ob

        self._ob = ob
        if self.camera_id:
            context = ob.Context()
            devices = context.query_devices()
            try:
                device = devices.get_device_by_serial_number(self.camera_id)
            except Exception as exc:  # noqa: BLE001 - SDK exception type varies by release
                raise RuntimeError(
                    f"Orbbec serial {self.camera_id!r} was not found or could not be opened"
                ) from exc
            actual_serial = str(device.get_device_info().get_serial_number())
            if actual_serial != self.camera_id:
                raise RuntimeError(
                    f"Orbbec serial mismatch: requested {self.camera_id!r}, got {actual_serial!r}"
                )
            self._context = context
            self._device = device
            self.device_identity = actual_serial
            pipe = ob.Pipeline(device)
        else:
            pipe = ob.Pipeline()
        cfg = ob.Config()
        color_profiles = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        requested = (self.width, self.height, self.fps)
        if all(value is not None for value in requested):
            requested_fps = float(self.fps)
            if not requested_fps.is_integer():
                raise RuntimeError("Orbbec capture FPS must be an integer SDK profile")
            format_name = self.color_format or "RGB"
            try:
                color_format = getattr(ob.OBFormat, format_name)
            except AttributeError as exc:
                raise RuntimeError(f"Orbbec color format {format_name!r} is unavailable") from exc
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    int(self.width), int(self.height), color_format, int(requested_fps),
                )
            except Exception as exc:  # noqa: BLE001 - SDK exception type varies by release
                raise RuntimeError(
                    "requested Orbbec color mode is unavailable: "
                    f"{self.width}x{self.height}@{requested_fps:g}"
                ) from exc
        elif any(value is not None for value in requested):
            raise RuntimeError("Orbbec width, height, and FPS must be specified together")
        else:
            color_profile = color_profiles.get_default_video_stream_profile()
        depth_profiles = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        if self.depth_mode is not None:
            try:
                depth_fps = float(self.depth_mode["fps"])
                if not depth_fps.is_integer():
                    raise ValueError("depth FPS is not integral")
                depth_format = getattr(ob.OBFormat, str(self.depth_mode["format"]))
                depth_profile = depth_profiles.get_video_stream_profile(
                    int(self.depth_mode["width"]), int(self.depth_mode["height"]),
                    depth_format, int(depth_fps),
                )
            except Exception as exc:  # noqa: BLE001 - SDK exception type varies by release
                raise RuntimeError(
                    f"requested Orbbec depth mode is unavailable: {self.depth_mode!r}"
                ) from exc
        else:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        align_filter = None
        if self.align:
            try:
                align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            except Exception as exc:  # noqa: BLE001 - SDK exception type varies by release
                if self.color_format is not None or self.depth_mode is not None:
                    raise RuntimeError("requested Orbbec software alignment is unavailable") from exc
        cfg.enable_stream(color_profile)
        cfg.enable_stream(depth_profile)
        try:
            active_video_profile = color_profile.as_video_stream_profile()
            active_intrinsics = active_video_profile.get_intrinsic()
            color_intrinsics = {
                "fx": float(active_intrinsics.fx),
                "fy": float(active_intrinsics.fy),
                "cx": float(active_intrinsics.cx),
                "cy": float(active_intrinsics.cy),
                "width": int(active_video_profile.get_width()),
                "height": int(active_video_profile.get_height()),
            }
        except Exception as exc:  # noqa: BLE001 - SDK profiles vary
            raise RuntimeError("active Orbbec color intrinsics are unavailable") from exc
        color_distortion = _active_orbbec_color_distortion(active_video_profile)
        if (
            color_intrinsics["width"] != int(color_profile.get_width())
            or color_intrinsics["height"] != int(color_profile.get_height())
            or not all(np.isfinite(color_intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
            or color_intrinsics["fx"] <= 0
            or color_intrinsics["fy"] <= 0
        ):
            raise RuntimeError("active Orbbec color intrinsics are invalid")
        try:
            cfg.set_frame_aggregate_output_mode(
                ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE,
            )
        except Exception as exc:  # noqa: BLE001 - required for paired RGB-D frames
            raise RuntimeError("Orbbec FULL_FRAME_REQUIRE aggregation is unavailable") from exc
        pipe.start(cfg)
        self._pipe = pipe
        self._align = align_filter
        self.color_intrinsics = color_intrinsics
        self.color_distortion = color_distortion
        self.negotiated_mode = {
            "width": int(color_profile.get_width()),
            "height": int(color_profile.get_height()),
            "fps": float(color_profile.get_fps()),
            "color_format": _sdk_format_name(color_profile.get_format()),
            "depth_mode": {
                "width": int(depth_profile.get_width()),
                "height": int(depth_profile.get_height()),
                "fps": float(depth_profile.get_fps()),
                "format": _sdk_format_name(depth_profile.get_format()),
            },
            "align_mode": "software_depth_to_color" if align_filter is not None else "disabled",
            "frame_aggregate_mode": "full_frame_require",
        }

    def _start_avfoundation(self) -> None:
        import cv2

        api = getattr(cv2, "CAP_AVFOUNDATION", 0)
        self._cap = cv2.VideoCapture(self.index, api) if api else cv2.VideoCapture(self.index)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            self._cap.release(); self._cap = None
            raise RuntimeError(f"camera index {self.index} would not open (permission/index/busy?)")
        self.negotiated_mode = {
            "width": int(round(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS)),
            "color_format": "opencv_decoded_bgr",
            "depth_mode": None,
            "align_mode": "disabled",
            "frame_aggregate_mode": "not_applicable",
        }

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release(); self._cap = None
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None
                self._device = None
                self._context = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- grab threads (keep latest frame only; never block readers) --------
    def _begin_inflight(self) -> tuple[int, float]:
        """Atomically timestamp and register one backend frame for End handoff."""

        with self._condition:
            if self._inflight_id is not None:
                raise RuntimeError("camera capture registered overlapping in-flight frames")
            self._next_inflight_id += 1
            inflight_id = self._next_inflight_id
            host_arrival_t = time.monotonic()
            self._inflight_id = inflight_id
            self._inflight_host_t = host_arrival_t
            return inflight_id, host_arrival_t

    def _complete_inflight(
        self,
        inflight_id: int,
        host_arrival_t: float,
        *,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        device_t_ms: float | None = None,
        time_source: str = "host_arrival",
    ) -> CameraFrame | None:
        """Publish a decoded frame and release any End waiter in one critical section."""

        with self._condition:
            published = None
            if rgb is not None or depth is not None:
                self._count += 1
                published = CameraFrame(
                    frame=rgb,
                    t=host_arrival_t,
                    index=self._count,
                    status=STATUS_OK,
                    depth=depth,
                    depth_scale=depth_scale,
                    device_t_ms=device_t_ms,
                    time_source=time_source,
                )
                self._latest = published
                # End owns a separate snapshot.  A later post-End publish may
                # replace the public one-slot buffer, but can never replace the
                # frame that belongs to the sealed interval.
                if (
                    self._end_boundary_target_id == inflight_id
                    and self._end_boundary_t is not None
                    and host_arrival_t <= self._end_boundary_t
                ):
                    self._end_boundary_snapshot = published

                if (
                    self._delivery_start_target_id == inflight_id
                    and self._delivery_start_t is not None
                    and host_arrival_t <= self._delivery_start_t
                ):
                    self._delivery_start_boundary_snapshot = published

                inside_delivery_interval = (
                    self._delivery_interval_active
                    and self._delivery_start_t is not None
                    and host_arrival_t > self._delivery_start_t
                    and (
                        self._end_boundary_t is None
                        or host_arrival_t <= self._end_boundary_t
                    )
                )
                on_recording_phase = (
                    (int(published.index) - self._delivery_phase_anchor)
                    % self.recording_frame_divisor == 0
                )
                if (
                    inside_delivery_interval
                    and self._delivery_accepting
                    and on_recording_phase
                ):
                    if len(self._delivery_queue) >= self.delivery_capacity_frames:
                        self._delivery_overflow_total += 1
                        self._delivery_interval_overflow += 1
                        if self._delivery_first_overflow_index is None:
                            self._delivery_first_overflow_index = int(published.index)
                        # Preserve the exact accepted prefix. Resuming after space
                        # becomes available would create an unrepresentable gap.
                        self._delivery_accepting = False
                    else:
                        self._delivery_queue.append(published)
                        self._delivery_enqueued += 1
                        if self._delivery_first_enqueued_index is None:
                            self._delivery_first_enqueued_index = int(published.index)
                        self._delivery_last_enqueued_index = int(published.index)
                        self._delivery_max_depth = max(
                            self._delivery_max_depth, len(self._delivery_queue),
                        )
                elif (
                    inside_delivery_interval
                    and on_recording_phase
                    and self._delivery_interval_overflow
                ):
                    self._delivery_rejected_after_overflow += 1

            if self._end_boundary_target_id == inflight_id:
                self._end_boundary_target_completed = True
                self._end_boundary_target_produced_frame = published is not None
            if self._delivery_start_target_id == inflight_id:
                self._delivery_start_target_completed = True
                self._delivery_start_target_produced_frame = published is not None
            if self._inflight_id == inflight_id:
                self._inflight_id = None
                self._inflight_host_t = None
            self._condition.notify_all()
            return published

    def begin_delivery_interval(self, *, timeout_s: float = 2.0) -> CameraStartBoundary:
        """Arm bounded lossless delivery after downstream files are ready.

        A frame already registered before Start is allowed to finish and becomes
        part of the boundary, not the recording. Frames registered afterward are
        queued immediately even while this caller waits for that decode to finish.
        """

        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("camera Start boundary timeout must be finite and positive")
        with self._condition:
            if self._delivery_interval_active or self._delivery_closing:
                raise RuntimeError("camera delivery interval is already active")
            self._delivery_discarded += len(self._delivery_queue)
            self._delivery_queue.clear()
            self._delivery_interval_generation += 1
            self._delivery_interval_id = self._delivery_interval_generation
            self._delivery_enqueued = 0
            self._delivery_delivered = 0
            self._delivery_interval_overflow = 0
            self._delivery_rejected_after_overflow = 0
            self._delivery_first_enqueued_index = None
            self._delivery_last_enqueued_index = None
            self._delivery_last_delivered_index = None
            self._delivery_first_overflow_index = None
            self._delivery_max_depth = 0
            self._end_boundary_t = None
            self._end_boundary_snapshot = None
            self._end_boundary_target_id = None
            self._end_boundary_target_completed = False
            self._end_boundary_target_produced_frame = None
            self._end_boundary_result = None

            self._delivery_start_t = time.monotonic()
            current = self._latest
            self._delivery_start_boundary_snapshot = (
                current
                if current is not None and current.t <= self._delivery_start_t
                else None
            )
            target = (
                self._inflight_id
                if (
                    self._inflight_id is not None
                    and self._inflight_host_t is not None
                    and self._inflight_host_t <= self._delivery_start_t
                )
                else None
            )
            self._delivery_start_target_id = target
            self._delivery_start_target_completed = target is None
            self._delivery_start_target_produced_frame = None
            self._delivery_interval_active = True
            self._delivery_accepting = True
            self._delivery_closing = False

            deadline = time.monotonic() + timeout
            while target is not None and not self._delivery_start_target_completed:
                if self._stop.is_set():
                    self._delivery_interval_active = False
                    self._delivery_accepting = False
                    raise RuntimeError(
                        "camera Start boundary stopped while waiting for an in-flight frame"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._delivery_interval_active = False
                    self._delivery_accepting = False
                    raise RuntimeError(
                        "camera Start boundary timed out waiting for an in-flight frame"
                    )
                self._condition.wait(timeout=remaining)

            snapshot = self._delivery_start_boundary_snapshot
            self._delivery_phase_anchor = (
                0 if snapshot is None else int(snapshot.index)
            ) + 1
            interval_id = int(self._delivery_interval_id)
            start_t = float(self._delivery_start_t)
            return CameraStartBoundary(
                interval_id=interval_id,
                t_start=start_t,
                frame=snapshot,
                proof={
                    "boundary_latched_atomically": True,
                    "interval_id": interval_id,
                    "start_boundary_frame_index": (
                        0 if snapshot is None else int(snapshot.index)
                    ),
                    "start_boundary_host_t": (
                        None if snapshot is None else float(snapshot.t)
                    ),
                    "inflight_pre_boundary_seen": target is not None,
                    "inflight_pre_boundary_completed": (
                        target is None or self._delivery_start_target_completed
                    ),
                    "inflight_pre_boundary_produced_frame": (
                        self._delivery_start_target_produced_frame
                        if target is not None else False
                    ),
                    "capacity": self.delivery_capacity_frames,
                    "overflow_baseline": self._delivery_overflow_total,
                    "recording_frame_divisor": self.recording_frame_divisor,
                    "phase_anchor_index": self._delivery_phase_anchor,
                },
            )

    def drain_delivery(
        self, *, max_frames: int | None = None, interval_id: int | None = None,
    ) -> tuple[CameraFrame, ...]:
        """Pop a consecutive batch without touching the latest preview slot."""

        if max_frames is not None and int(max_frames) <= 0:
            raise ValueError("camera delivery batch size must be positive")
        with self._condition:
            if (
                interval_id is not None
                and self._delivery_interval_id is not None
                and int(interval_id) != self._delivery_interval_id
            ):
                raise RuntimeError("camera delivery interval id is stale")
            limit = (
                len(self._delivery_queue)
                if max_frames is None
                else min(len(self._delivery_queue), int(max_frames))
            )
            batch = tuple(self._delivery_queue.popleft() for _ in range(limit))
            if batch:
                self._delivery_delivered += len(batch)
                self._delivery_last_delivered_index = int(batch[-1].index)
            return batch

    def delivery_health(self) -> dict[str, object]:
        with self._condition:
            now = time.monotonic()
            thread_alive = self._thread is not None and self._thread.is_alive()
            last_frame_age_s = (
                None if self._latest is None
                else max(0.0, now - float(self._latest.t))
            )
            freshness_limit_s = self._freshness_limit_s()
            frame_fresh = (
                last_frame_age_s is not None
                and last_frame_age_s <= freshness_limit_s
            )
            if self._thread_error is not None:
                status = "ERROR"
            elif self._thread is not None and not thread_alive:
                status = "STOPPED"
            elif self._delivery_interval_overflow:
                status = "OVERFLOW"
            elif self._latest is not None and not frame_fresh:
                status = "STALE"
            elif self._latest is None:
                status = "NO_DATA"
            else:
                status = "OK"
            return {
                "status": status,
                "thread_alive": thread_alive,
                "thread_error": self._thread_error,
                "last_frame_index": (
                    None if self._latest is None else int(self._latest.index)
                ),
                "last_frame_host_t": (
                    None if self._latest is None else float(self._latest.t)
                ),
                "last_frame_age_s": last_frame_age_s,
                "freshness_limit_s": freshness_limit_s,
                "frame_fresh": frame_fresh,
                "capacity": self.delivery_capacity_frames,
                "interval_id": self._delivery_interval_id,
                "interval_active": self._delivery_interval_active,
                "accepting": self._delivery_accepting,
                "closing": self._delivery_closing,
                "queue_depth": len(self._delivery_queue),
                "max_queue_depth": self._delivery_max_depth,
                "interval_enqueued_frames": self._delivery_enqueued,
                "interval_delivered_frames": self._delivery_delivered,
                "first_enqueued_frame_index": self._delivery_first_enqueued_index,
                "last_enqueued_frame_index": self._delivery_last_enqueued_index,
                "last_delivered_frame_index": self._delivery_last_delivered_index,
                "overflow_frames": self._delivery_overflow_total,
                "interval_overflow_frames": self._delivery_interval_overflow,
                "first_overflow_frame_index": self._delivery_first_overflow_index,
                "rejected_after_overflow": self._delivery_rejected_after_overflow,
                "discarded_frames": self._delivery_discarded,
            }

    def _freshness_limit_s(self) -> float:
        mode = self.negotiated_mode
        raw_fps = mode.get("fps") if isinstance(mode, dict) else self.fps
        try:
            fps = float(raw_fps if raw_fps is not None else 30.0)
        except (TypeError, ValueError):
            fps = 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        return 3.0 / fps

    def _assert_live_boundary_locked(
        self, *, boundary_t: float, label: str, snapshot=...,
    ) -> None:
        if self._thread_error is not None:
            raise RuntimeError(f"camera {label} boundary capture failed: {self._thread_error}")
        # Direct primitive tests exercise boundary mechanics before start(). Once
        # a capture thread exists, however, its liveness is mandatory.
        if self._thread is not None and not self._thread.is_alive():
            raise RuntimeError(f"camera {label} boundary capture thread is not alive")
        if snapshot is ...:
            snapshot = self._latest
        if snapshot is None:
            raise RuntimeError(f"camera {label} boundary has no frame snapshot")
        age_s = max(0.0, float(boundary_t) - float(snapshot.t))
        if age_s > self._freshness_limit_s():
            raise RuntimeError(
                f"camera {label} boundary snapshot is stale: "
                f"{age_s:.6f}s > {self._freshness_limit_s():.6f}s"
            )

    def abort_delivery_interval(self) -> dict[str, object]:
        """Close and discard only an already-invalid/unsealed interval."""

        with self._condition:
            self._delivery_interval_active = False
            self._delivery_accepting = False
            self._delivery_closing = False
            self._delivery_discarded += len(self._delivery_queue)
            self._delivery_queue.clear()
            self._condition.notify_all()
        return self.delivery_health()

    def _run_orbbec(self) -> None:
        ob = self._ob
        while not self._stop.is_set():
            fs = self._pipe.wait_for_frames(100)
            if fs is None:
                continue
            inflight_id, host_arrival_t = self._begin_inflight()
            rgb = depth = None
            depth_scale = 1.0
            device_t_ms = None
            try:
                if self._align is not None:
                    fs = _as_orbbec_frameset(self._align.process(fs))
                else:
                    fs = _as_orbbec_frameset(fs)
                if fs is not None:
                    cf, df = fs.get_color_frame(), fs.get_depth_frame()
                    rgb = _ob_color_to_rgb(cf, ob) if cf is not None else None
                    if df is not None:
                        depth = _ob_depth_to_array(df)
                        raw_depth_scale = (
                            df.get_depth_scale()
                            if hasattr(df, "get_depth_scale")
                            else 1.0
                        )
                        depth_scale = (
                            1.0 if raw_depth_scale is None else float(raw_depth_scale)
                        )
                    device_t_ms = _orbbec_device_time_ms(cf)
            finally:
                self._complete_inflight(
                    inflight_id,
                    host_arrival_t,
                    rgb=rgb,
                    depth=depth,
                    depth_scale=depth_scale,
                    device_t_ms=device_t_ms,
                    time_source="orbbec_device_and_host_arrival",
                )

    def _run_fake(self) -> None:
        """Synthetic RGB + depth source so the full RGB+depth path (CameraStream + recorder
        + consumer) can be validated on any machine without the Orbbec hardware."""
        h, w = (self.height or 480), (self.width or 640)
        yy, xx = np.mgrid[0:h, 0:w]
        radial = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        period_s = 1.0 / float(self.fps or 30.0)
        next_frame_t = time.monotonic()
        while not self._stop.is_set():
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]   # x gradient
            rgb[:, :, 1] = np.uint8(((self._count + 1) * 5) % 256)            # time-varying
            rgb[:, :, 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]    # y gradient
            depth = (500 + radial * 2.0).astype(np.uint16)                    # mm, radial
            inflight_id, host_arrival_t = self._begin_inflight()
            self._complete_inflight(
                inflight_id,
                host_arrival_t,
                rgb=rgb if self.rgb else rgb[..., ::-1],
                depth=depth,
                time_source="synthetic_host",
            )
            # Schedule against an absolute deadline so array generation and
            # publication time do not silently lower the declared frame rate.
            # If the process falls more than one whole period behind, rebase
            # instead of emitting an unbounded catch-up burst.
            next_frame_t += period_s
            now = time.monotonic()
            if next_frame_t < now - period_s:
                next_frame_t = now
            time.sleep(max(0.0, next_frame_t - now))

    def _run_cv2(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            inflight_id, host_arrival_t = self._begin_inflight()
            rgb = None
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame
            finally:
                self._complete_inflight(
                    inflight_id,
                    host_arrival_t,
                    rgb=rgb,
                    time_source="opencv_host_arrival",
                )

    def latch_end_boundary(
        self, *, timeout_s: float = 2.0, interval_id: int | None = None,
    ) -> CameraEndBoundary:
        """Freeze and return the last decoded frame whose host time is at or before End.

        End is sampled while holding the same condition used for frame-arrival
        timestamp registration.  If that frame is still in alignment/decode,
        this waits on the condition until it completes; polling sleeps are not
        used and a timeout is fail-closed.
        """

        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("camera End boundary timeout must be finite and positive")
        with self._condition:
            if (
                interval_id is not None
                and self._delivery_interval_id is not None
                and int(interval_id) != self._delivery_interval_id
            ):
                raise RuntimeError("camera End boundary interval id is stale")
            if self._end_boundary_result is not None:
                return self._end_boundary_result
            if self._end_boundary_t is None:
                proposed_end_t = time.monotonic()
                if self._thread_error is not None:
                    raise RuntimeError(
                        f"camera End boundary capture failed: {self._thread_error}"
                    )
                if self._thread is not None and not self._thread.is_alive():
                    raise RuntimeError(
                        "camera End boundary capture thread is not alive"
                    )
                self._end_boundary_t = proposed_end_t
                self._delivery_closing = self._delivery_interval_active
                current = self._latest
                self._end_boundary_snapshot = (
                    current
                    if current is not None and current.t <= self._end_boundary_t
                    else None
                )
                target = (
                    self._inflight_id
                    if (
                        self._inflight_id is not None
                        and self._inflight_host_t is not None
                        and self._inflight_host_t <= self._end_boundary_t
                    )
                    else None
                )
                self._end_boundary_target_id = target
                self._end_boundary_target_completed = target is None
                self._end_boundary_target_produced_frame = None

            deadline = time.monotonic() + timeout
            while (
                self._end_boundary_target_id is not None
                and not self._end_boundary_target_completed
            ):
                if self._stop.is_set():
                    raise RuntimeError(
                        "camera End boundary stopped while waiting for an in-flight "
                        "pre-boundary frame"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "camera End boundary timed out waiting for an in-flight "
                        "pre-boundary frame"
                    )
                self._condition.wait(timeout=remaining)

            snapshot = self._end_boundary_snapshot
            self._assert_live_boundary_locked(
                boundary_t=float(self._end_boundary_t), label="End", snapshot=snapshot,
            )
            target_seen = self._end_boundary_target_id is not None
            self._delivery_interval_active = False
            self._delivery_accepting = False
            self._delivery_closing = False
            result = CameraEndBoundary(
                t_end=self._end_boundary_t,
                frame=snapshot,
                proof={
                    "boundary_latched_atomically": True,
                    "inflight_pre_boundary_seen": target_seen,
                    "inflight_pre_boundary_completed": (
                        not target_seen or self._end_boundary_target_completed
                    ),
                    "inflight_pre_boundary_produced_frame": (
                        self._end_boundary_target_produced_frame
                        if target_seen
                        else False
                    ),
                    "post_boundary_overwrite_prevented": True,
                    "snapshot_frame_index": (
                        None if snapshot is None else int(snapshot.index)
                    ),
                    "snapshot_host_t": None if snapshot is None else float(snapshot.t),
                    "snapshot_age_s": (
                        None if snapshot is None
                        else max(0.0, float(self._end_boundary_t) - float(snapshot.t))
                    ),
                    "max_snapshot_age_s": self._freshness_limit_s(),
                    "thread_alive_at_boundary": (
                        self._thread is None or self._thread.is_alive()
                    ),
                    "thread_error_at_boundary": self._thread_error,
                    "interval_id": self._delivery_interval_id,
                    "delivery_capacity": self.delivery_capacity_frames,
                    "delivery_queue_depth": len(self._delivery_queue),
                    "delivery_enqueued_frames": self._delivery_enqueued,
                    "delivery_delivered_frames": self._delivery_delivered,
                    "delivery_first_frame_index": self._delivery_first_enqueued_index,
                    "delivery_last_frame_index": self._delivery_last_enqueued_index,
                    "delivery_overflow_frames": self._delivery_interval_overflow,
                },
            )
            self._end_boundary_result = result
            return result

    # -- non-blocking read -------------------------------------------------
    def latest_frame(self) -> CameraFrame:
        with self._lock:
            s = self._latest
        now = time.monotonic()
        if s is None:
            return CameraFrame(frame=None, t=now, index=0, status=STATUS_NO_DATA)
        age = max(0.0, now - s.t)
        status = STATUS_STALE if age > self.stale_timeout_s else STATUS_OK
        return CameraFrame(frame=s.frame, t=s.t, index=s.index, status=status,
                           depth=s.depth, depth_scale=s.depth_scale, age_s=age,
                           device_t_ms=s.device_t_ms, time_source=s.time_source)


def preview(backend: str = "orbbec", index: int = 0, show_depth: bool = True) -> None:
    """Live window for aiming the camera (RGB, plus colorized depth if available)."""
    import cv2

    print(f"live preview ({backend}) — aim it, press 'q'/Esc in the window to quit")
    with CameraStream(backend=backend, index=index, rgb=False) as cam:  # BGR for cv2.imshow
        win = f"camera preview ({backend})"
        while True:
            f = cam.latest_frame()
            panels = []
            if f.frame is not None:
                img = f.frame.copy()
                if backend == "orbbec":  # frame is RGB here -> back to BGR for imshow
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, f"{img.shape[1]}x{img.shape[0]} #{f.index} {f.status}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                panels.append(img)
            if show_depth and f.depth is not None:
                d = f.depth.astype(np.float32)
                d = np.clip(d * f.depth_scale, 0, 4000) / 4000.0 * 255.0
                dcol = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)
                if panels and dcol.shape[:2] != panels[0].shape[:2]:
                    dcol = cv2.resize(dcol, (panels[0].shape[1], panels[0].shape[0]))
                panels.append(dcol)
            if panels:
                cv2.imshow(win, np.hstack(panels) if len(panels) > 1 else panels[0])
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":  # probe: camera_stream.py [orbbec|avf] [index]  | preview: ... preview ...
    import sys

    args = sys.argv[1:]
    if args and args[0] == "preview":
        bk = args[1] if len(args) > 1 else "orbbec"
        preview(backend=bk, index=int(args[2]) if len(args) > 2 else 0)
    else:
        bk = args[0] if args else "orbbec"
        idx = int(args[1]) if len(args) > 1 else 0
        try:
            with CameraStream(backend=bk, index=idx) as cam:
                time.sleep(1.5)
                f = cam.latest_frame()
                rgb = None if f.frame is None else f.frame.shape
                dep = None if f.depth is None else f.depth.shape
                print(f"{bk}: status={f.status} rgb={rgb} depth={dep} scale={f.depth_scale} grabbed={f.index}")
        except Exception as e:  # noqa: BLE001
            print(f"{bk}: {type(e).__name__}: {e}")
