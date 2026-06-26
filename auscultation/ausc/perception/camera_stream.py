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
from dataclasses import dataclass
from typing import Optional

import numpy as np

STATUS_OK = "OK"
STATUS_STALE = "STALE"
STATUS_NO_DATA = "NO_DATA"


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


def _ob_color_to_rgb(color_frame, ob) -> Optional[np.ndarray]:
    """Decode an Orbbec color frame to an RGB uint8 image, covering common formats."""
    import cv2

    w, h = color_frame.get_width(), color_frame.get_height()
    buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    F = ob.OBFormat
    fmt = color_frame.get_format()

    def is_(name: str) -> bool:  # tolerate SDK builds that lack a given format constant
        return hasattr(F, name) and fmt == getattr(F, name)

    try:
        if is_("RGB"):
            return buf.reshape(h, w, 3)
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


class CameraStream:
    """Background camera capture (Orbbec RGB+depth, or OpenCV RGB). Open with `start()`
    (or `with`), then call the non-blocking `latest_frame()` from any loop."""

    def __init__(
        self,
        backend: str = "orbbec",
        index: int = 0,
        align_depth_to_color: bool = True,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        rgb: bool = True,
        stale_timeout_ms: float = 500.0,
    ) -> None:
        self.backend = backend
        self.index = int(index)
        self.align = align_depth_to_color
        self.width = width
        self.height = height
        self.fps = fps
        self.rgb = rgb
        self.stale_timeout_s = stale_timeout_ms / 1000.0
        self._cap = None          # cv2 VideoCapture (avfoundation)
        self._pipe = None         # pyorbbecsdk Pipeline (orbbec)
        self._ob = None
        self._align = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._count = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "CameraStream":
        if self.backend == "orbbec":
            self._start_orbbec(); target = self._run_orbbec
        elif self.backend == "fake":            # synthetic RGB+depth (no hardware; for tests/CI)
            target = self._run_fake
        else:
            self._start_avfoundation(); target = self._run_cv2
        self._stop.clear()
        self._thread = threading.Thread(target=target, name="camera-stream", daemon=True)
        self._thread.start()
        return self

    def _start_orbbec(self) -> None:
        import pyorbbecsdk as ob

        self._ob = ob
        pipe = ob.Pipeline()
        cfg = ob.Config()
        for sensor in (ob.OBSensorType.COLOR_SENSOR, ob.OBSensorType.DEPTH_SENSOR):
            prof = pipe.get_stream_profile_list(sensor).get_default_video_stream_profile()
            cfg.enable_stream(prof)
        pipe.start(cfg)
        self._pipe = pipe
        if self.align:
            try:
                self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
            except Exception:  # noqa: BLE001 - alignment optional
                self._align = None

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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release(); self._cap = None
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- grab threads (keep latest frame only; never block readers) --------
    def _run_orbbec(self) -> None:
        ob = self._ob
        while not self._stop.is_set():
            fs = self._pipe.wait_for_frames(100)
            if fs is None:
                continue
            if self._align is not None:
                fs = self._align.process(fs) or fs
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            rgb = _ob_color_to_rgb(cf, ob) if cf is not None else None
            depth = scale = None
            if df is not None:
                depth = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(
                    df.get_height(), df.get_width())
                scale = df.get_depth_scale() if hasattr(df, "get_depth_scale") else 1.0
            if rgb is None and depth is None:
                continue
            self._count += 1
            s = CameraFrame(frame=rgb, t=time.monotonic(), index=self._count, status=STATUS_OK,
                            depth=depth, depth_scale=scale if scale is not None else 1.0)
            with self._lock:
                self._latest = s

    def _run_fake(self) -> None:
        """Synthetic RGB + depth source so the full RGB+depth path (CameraStream + recorder
        + consumer) can be validated on any machine without the Orbbec hardware."""
        h, w = (self.height or 480), (self.width or 640)
        yy, xx = np.mgrid[0:h, 0:w]
        radial = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        while not self._stop.is_set():
            self._count += 1
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]   # x gradient
            rgb[:, :, 1] = np.uint8((self._count * 5) % 256)                  # time-varying
            rgb[:, :, 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]    # y gradient
            depth = (500 + radial * 2.0).astype(np.uint16)                    # mm, radial
            s = CameraFrame(frame=rgb if self.rgb else rgb[..., ::-1], t=time.monotonic(),
                            index=self._count, status=STATUS_OK, depth=depth, depth_scale=1.0)
            with self._lock:
                self._latest = s
            time.sleep(1.0 / 30.0)

    def _run_cv2(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._count += 1
            s = CameraFrame(frame=frame, t=time.monotonic(), index=self._count, status=STATUS_OK)
            with self._lock:
                self._latest = s

    # -- non-blocking read -------------------------------------------------
    def latest_frame(self) -> CameraFrame:
        now = time.monotonic()
        with self._lock:
            s = self._latest
        if s is None:
            return CameraFrame(frame=None, t=now, index=0, status=STATUS_NO_DATA)
        age = now - s.t
        status = STATUS_STALE if age > self.stale_timeout_s else STATUS_OK
        return CameraFrame(frame=s.frame, t=s.t, index=s.index, status=status,
                           depth=s.depth, depth_scale=s.depth_scale, age_s=age)


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
