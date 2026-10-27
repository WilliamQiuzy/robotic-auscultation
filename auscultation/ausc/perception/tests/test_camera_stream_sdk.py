"""Deployment camera-ingest tests (no hardware: Orbbec SDK + frames are mocked).

camera_stream.py is the live RGB+depth ingest feeding the detector — bugs here mean garbage frames
reach the pipeline. Previously untested. Covered here:
  - _ob_color_to_rgb: every supported color format + unknown/malformed (must fail to None, not crash)
  - CameraStream._orbbec_frame: the per-frameset ingest logic (color/depth/scale/alignment/skip)
  - latest_frame: the non-blocking freshness contract (NO_DATA / OK / STALE)
  - the threaded lifecycle via the built-in synthetic "fake" backend
"""
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import cv2  # noqa: E402
import pytest  # noqa: E402

from ausc.perception.camera_stream import (  # noqa: E402
    CameraFrame, CameraStream, _ob_color_to_rgb,
    STATUS_NO_DATA, STATUS_OK, STATUS_STALE,
)


# ── fake Orbbec SDK surface ───────────────────────────────────────────────────
class _Fmt:
    RGB = "RGB"; BGR = "BGR"; MJPG = "MJPG"; YUYV = "YUYV"; UYVY = "UYVY"; NV12 = "NV12"


class _FmtPartial:           # an SDK build missing some format constants (is_ must tolerate this)
    RGB = "RGB"


class _Ob:
    OBFormat = _Fmt


class _Color:
    def __init__(self, w, h, data, fmt):
        self._w, self._h, self._data, self._fmt = w, h, data, fmt

    def get_width(self): return self._w
    def get_height(self): return self._h
    def get_data(self): return self._data
    def get_format(self): return self._fmt


class _Depth:
    def __init__(self, arr, scale=0.5):
        self._arr = np.asarray(arr, np.uint16); self._scale = scale

    def get_data(self): return self._arr.tobytes()
    def get_height(self): return self._arr.shape[0]
    def get_width(self): return self._arr.shape[1]
    def get_depth_scale(self): return self._scale


class _DepthBare:
    """A depth frame whose SDK build truly lacks get_depth_scale (scale must default to 1.0)."""
    def __init__(self, arr):
        self._arr = np.asarray(arr, np.uint16)

    def get_data(self): return self._arr.tobytes()
    def get_height(self): return self._arr.shape[0]
    def get_width(self): return self._arr.shape[1]


class _FS:
    def __init__(self, color=None, depth=None):
        self._c, self._d = color, depth

    def get_color_frame(self): return self._c
    def get_depth_frame(self): return self._d


def _color(img_or_buf, fmt, w, h):
    data = img_or_buf.tobytes() if isinstance(img_or_buf, np.ndarray) else img_or_buf
    return _Color(w, h, data, fmt)


# ── _ob_color_to_rgb: format coverage ─────────────────────────────────────────
def test_color_rgb_passthrough_exact():
    img = np.random.RandomState(0).randint(0, 256, (8, 10, 3), np.uint8)
    out = _ob_color_to_rgb(_color(img, _Fmt.RGB, 10, 8), _Ob())
    assert out.shape == (8, 10, 3) and np.array_equal(out, img)


def test_color_bgr_is_channel_swapped():
    bgr = np.random.RandomState(1).randint(0, 256, (8, 10, 3), np.uint8)
    out = _ob_color_to_rgb(_color(bgr, _Fmt.BGR, 10, 8), _Ob())
    assert np.array_equal(out, bgr[:, :, ::-1])


def test_color_mjpg_decodes():
    bgr = np.random.RandomState(2).randint(0, 256, (16, 16, 3), np.uint8)
    ok, jpg = cv2.imencode(".jpg", bgr)
    assert ok
    out = _ob_color_to_rgb(_color(jpg.flatten(), _Fmt.MJPG, 16, 16), _Ob())
    assert out is not None and out.shape == (16, 16, 3)


def test_color_yuyv_uyvy_nv12_shapes():
    for fmt, nbytes_factor, rows in ((_Fmt.YUYV, 2, 8), (_Fmt.UYVY, 2, 8)):
        buf = np.zeros(8 * 10 * nbytes_factor, np.uint8)
        out = _ob_color_to_rgb(_color(buf, fmt, 10, 8), _Ob())
        assert out is not None and out.shape == (8, 10, 3)
    nv12 = np.zeros((8 * 3 // 2) * 10, np.uint8)
    out = _ob_color_to_rgb(_color(nv12, _Fmt.NV12, 10, 8), _Ob())
    assert out is not None and out.shape == (8, 10, 3)


def test_color_unknown_format_returns_none():
    buf = np.zeros(8 * 10 * 3, np.uint8)
    assert _ob_color_to_rgb(_color(buf, "WEIRD_FMT", 10, 8), _Ob()) is None


def test_color_malformed_buffer_returns_none_not_crash():
    # RGB format but the buffer is the wrong size → reshape raises → caught → None
    buf = np.zeros(8 * 10 * 3 - 5, np.uint8)
    assert _ob_color_to_rgb(_color(buf, _Fmt.RGB, 10, 8), _Ob()) is None


def test_color_missing_format_constant_tolerated():
    # SDK build lacks YUYV etc.; a YUYV frame is simply unknown → None, no AttributeError
    class _ObPartial:
        OBFormat = _FmtPartial
    buf = np.zeros(8 * 10 * 2, np.uint8)
    assert _ob_color_to_rgb(_color(buf, "YUYV", 10, 8), _ObPartial()) is None


# ── CameraStream._orbbec_frame: ingest logic ──────────────────────────────────
def _stream(align=None):
    cam = CameraStream(backend="orbbec")
    cam._ob = _Ob()
    cam._align = align
    return cam


def test_orbbec_frame_none_fs_is_skipped():
    assert _stream()._orbbec_frame(None) is None


def test_orbbec_frame_color_only():
    cam = _stream()
    img = np.random.RandomState(3).randint(0, 256, (8, 10, 3), np.uint8)
    cf = cam._orbbec_frame(_FS(color=_color(img, _Fmt.RGB, 10, 8)))
    assert cf is not None and cf.depth is None
    assert np.array_equal(cf.frame, img) and cf.status == STATUS_OK


def test_orbbec_frame_depth_only_keeps_scale():
    cam = _stream()
    d = np.arange(8 * 10).reshape(8, 10)
    cf = cam._orbbec_frame(_FS(depth=_Depth(d, scale=0.25)))
    assert cf.frame is None and cf.depth.shape == (8, 10)
    assert cf.depth.dtype == np.uint16 and np.array_equal(cf.depth, d.astype(np.uint16))
    assert cf.depth_scale == 0.25


def test_orbbec_frame_depth_without_get_scale_defaults_to_one():
    cam = _stream()
    cf = cam._orbbec_frame(_FS(depth=_DepthBare(np.ones((4, 4)))))
    assert cf is not None and cf.depth_scale == 1.0


def test_orbbec_frame_color_and_depth():
    cam = _stream()
    img = np.random.RandomState(4).randint(0, 256, (6, 6, 3), np.uint8)
    d = (np.arange(36).reshape(6, 6) * 7)
    cf = cam._orbbec_frame(_FS(color=_color(img, _Fmt.RGB, 6, 6), depth=_Depth(d, 1.0)))
    assert np.array_equal(cf.frame, img) and np.array_equal(cf.depth, d.astype(np.uint16))


def test_orbbec_frame_unusable_is_skipped():
    # color present but unknown format → rgb None; no depth → nothing usable → skip
    cam = _stream()
    cf = cam._orbbec_frame(_FS(color=_color(np.zeros(48, np.uint8), "WEIRD", 4, 4)))
    assert cf is None


def test_orbbec_frame_index_increments_only_on_kept_frames():
    cam = _stream()
    img = np.random.RandomState(5).randint(0, 256, (4, 4, 3), np.uint8)
    a = cam._orbbec_frame(_FS(color=_color(img, _Fmt.RGB, 4, 4)))
    cam._orbbec_frame(None)                                   # skipped → no increment
    cam._orbbec_frame(_FS(color=_color(np.zeros(48, np.uint8), "WEIRD", 4, 4)))  # skipped
    b = cam._orbbec_frame(_FS(color=_color(img, _Fmt.RGB, 4, 4)))
    assert a.index == 1 and b.index == 2


def test_orbbec_frame_alignment_is_applied():
    img = np.random.RandomState(6).randint(0, 256, (4, 4, 3), np.uint8)
    aligned = _FS(color=_color(img, _Fmt.RGB, 4, 4), depth=_Depth(np.ones((4, 4)), 1.0))

    class _Align:
        def __init__(self): self.called = False
        def process(self, fs): self.called = True; return aligned

    cam = _stream(align=_Align())
    cf = cam._orbbec_frame(_FS())                             # raw has nothing; align swaps in `aligned`
    assert cam._align.called and cf is not None and np.array_equal(cf.frame, img)


def test_orbbec_frame_alignment_none_falls_back_to_raw():
    img = np.random.RandomState(7).randint(0, 256, (4, 4, 3), np.uint8)
    raw = _FS(color=_color(img, _Fmt.RGB, 4, 4))

    class _AlignNull:
        def process(self, fs): return None                   # align failed → keep raw (the `or fs`)

    cam = _stream(align=_AlignNull())
    cf = cam._orbbec_frame(raw)
    assert cf is not None and np.array_equal(cf.frame, img)


# ── latest_frame: freshness contract ──────────────────────────────────────────
def test_latest_frame_no_data_before_first_grab():
    cam = CameraStream(backend="fake")
    f = cam.latest_frame()
    assert f.status == STATUS_NO_DATA and f.frame is None and f.depth is None


def test_latest_frame_ok_then_stale():
    cam = CameraStream(backend="fake", stale_timeout_ms=100.0)
    img = np.zeros((2, 2, 3), np.uint8)
    cam._latest = CameraFrame(frame=img, t=time.monotonic(), index=1, status=STATUS_OK)
    assert cam.latest_frame().status == STATUS_OK
    cam._latest = CameraFrame(frame=img, t=time.monotonic() - 10.0, index=1, status=STATUS_OK)
    out = cam.latest_frame()
    assert out.status == STATUS_STALE and out.age_s > 0.1


# ── threaded lifecycle via the synthetic 'fake' backend (no hardware) ──────────
def test_fake_backend_produces_frames_and_stops_cleanly():
    cam = CameraStream(backend="fake", width=64, height=48)
    with cam:
        f = None
        for _ in range(200):
            f = cam.latest_frame()
            if f.status == STATUS_OK and f.frame is not None:
                break
            time.sleep(0.005)
        assert f is not None and f.status == STATUS_OK
        assert f.frame.shape == (48, 64, 3) and f.depth.shape == (48, 64)
        assert f.depth_scale == 1.0 and f.index >= 1
    assert cam._thread is not None and not cam._thread.is_alive()    # stop() joined the thread


def test_cameraframe_is_immutable():
    f = CameraFrame(frame=None, t=0.0, index=0, status=STATUS_NO_DATA)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.t = 1.0
