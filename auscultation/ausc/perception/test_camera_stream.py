"""Camera timestamp contract tests without camera hardware."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import camera_stream as camera


class _Frame:
    def __init__(self, value):
        self.value = value

    def get_timestamp(self):
        return self.value


def _distortion(**overrides):
    values = {
        "k1": 0.01, "k2": 0.02, "k3": 0.03, "k4": 0.04,
        "k5": 0.05, "k6": 0.06, "p1": 0.07, "p2": 0.08,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_orbbec_device_timestamp_is_preserved_in_milliseconds():
    assert camera._orbbec_device_time_ms(_Frame(1234.5)) == 1234.5


def test_invalid_or_missing_device_timestamp_is_not_fabricated():
    assert camera._orbbec_device_time_ms(object()) is None
    assert camera._orbbec_device_time_ms(_Frame(float("nan"))) is None


def test_align_filter_generic_frame_is_converted_back_to_frameset():
    frameset = SimpleNamespace(get_color_frame=lambda: object(), get_depth_frame=lambda: object())
    generic_frame = SimpleNamespace(as_frame_set=lambda: frameset)
    assert camera._as_orbbec_frameset(generic_frame) is frameset
    assert camera._as_orbbec_frameset(frameset) is frameset


def test_orbbec_rgb_and_depth_arrays_own_their_sdk_buffer_bytes():
    rgb_bytes = bytearray([1, 2, 3, 4, 5, 6])
    depth_bytes = bytearray(np.asarray([100, 200], dtype=np.uint16).tobytes())
    rgb_frame = SimpleNamespace(
        get_width=lambda: 2,
        get_height=lambda: 1,
        get_data=lambda: rgb_bytes,
        get_format=lambda: "rgb",
    )
    depth_frame = SimpleNamespace(
        get_width=lambda: 2,
        get_height=lambda: 1,
        get_data=lambda: depth_bytes,
    )
    sdk = SimpleNamespace(OBFormat=SimpleNamespace(RGB="rgb"))

    rgb = camera._ob_color_to_rgb(rgb_frame, sdk)
    depth = camera._ob_depth_to_array(depth_frame)
    rgb_bytes[:] = b"\x09" * len(rgb_bytes)
    depth_bytes[:] = b"\x00" * len(depth_bytes)

    assert rgb.tolist() == [[[1, 2, 3], [4, 5, 6]]]
    assert depth.tolist() == [[100, 200]]
    assert rgb.flags.owndata is True
    assert depth.flags.owndata is True


def test_latest_frame_age_is_never_negative_if_callback_updates_during_read(monkeypatch):
    stream = camera.CameraStream(backend="fake")
    stream._latest = camera.CameraFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8), t=10.0, index=1, status="OK",
    )
    monkeypatch.setattr(camera.time, "monotonic", lambda: 9.99)
    assert stream.latest_frame().age_s == 0.0


def test_fake_camera_generation_period_matches_its_negotiated_fps(monkeypatch):
    stream = camera.CameraStream(
        backend="fake", width=8, height=6, fps=17,
    )
    sleeps = []

    def stop_after_one(period):
        sleeps.append(period)
        stream._stop.set()

    monkeypatch.setattr(camera.time, "sleep", stop_after_one)
    stream._run_fake()

    assert sleeps == [pytest.approx(1.0 / 17.0, abs=0.005)]


def test_start_cleans_up_a_partially_opened_orbbec_pipeline(monkeypatch):
    stopped = []
    stream = camera.CameraStream(backend="orbbec")

    def fail_after_pipeline_opened():
        stream._pipe = SimpleNamespace(stop=lambda: stopped.append(True))
        raise RuntimeError("injected profile failure")

    monkeypatch.setattr(stream, "_start_orbbec", fail_after_pipeline_opened)

    with pytest.raises(RuntimeError, match="injected profile failure"):
        stream.start()

    assert stopped == [True]
    assert stream._pipe is None


@pytest.mark.parametrize(
    "profile",
    [
        object(),
        SimpleNamespace(get_distortion=lambda: SimpleNamespace(k1=0.01)),
        SimpleNamespace(get_distortion=lambda: _distortion(k5=float("nan"))),
        SimpleNamespace(get_distortion=lambda: _distortion(p2=float("inf"))),
    ],
)
def test_active_orbbec_color_distortion_fails_closed_when_missing_or_nonfinite(profile):
    with pytest.raises(RuntimeError, match="active Orbbec color distortion"):
        camera._active_orbbec_color_distortion(profile)


def test_orbbec_open_binds_serial_and_exact_color_profile(monkeypatch):
    calls = {}

    class Profile:
        def __init__(self, width, height, fps, pixel_format):
            self.values = width, height, fps
            self.pixel_format = pixel_format

        def get_width(self): return self.values[0]
        def get_height(self): return self.values[1]
        def get_fps(self): return self.values[2]
        def get_format(self): return self.pixel_format
        def as_video_stream_profile(self): return self
        def get_intrinsic(self):
            return SimpleNamespace(fx=601.0, fy=602.0, cx=319.5, cy=239.5)
        def get_distortion(self):
            calls["active_color_distortion_queries"] = (
                calls.get("active_color_distortion_queries", 0) + 1
            )
            return SimpleNamespace(
                k1=0.01, k2=0.02, k3=0.03, k4=0.04,
                k5=0.05, k6=0.06, p1=0.07, p2=0.08,
            )

    color_profile = Profile(640, 480, 30, "RGB")
    depth_profile = Profile(320, 240, 30, "Y16")

    class Profiles:
        def __init__(self, sensor): self.sensor = sensor

        def get_video_stream_profile(self, width, height, pixel_format, fps):
            calls.setdefault("requested_profiles", []).append(
                (self.sensor, width, height, pixel_format, fps)
            )
            return depth_profile if self.sensor == "depth" else color_profile

        def get_default_video_stream_profile(self):
            calls.setdefault("default_profiles", []).append(self.sensor)
            return depth_profile if self.sensor == "depth" else color_profile

    class Device:
        def get_device_info(self):
            return SimpleNamespace(get_serial_number=lambda: "FEMTO-001")

    device = Device()

    class DeviceList:
        def get_device_by_serial_number(self, serial):
            calls["serial"] = serial
            return device

    class Context:
        def query_devices(self): return DeviceList()

    class Pipeline:
        def __init__(self, selected_device=None):
            calls["pipeline_device"] = selected_device

        def get_stream_profile_list(self, sensor): return Profiles(sensor)
        def start(self, config): calls["started"] = list(config.enabled)
        def stop(self): pass

    class Config:
        def __init__(self): self.enabled = []
        def enable_stream(self, profile): self.enabled.append(profile)
        def set_frame_aggregate_output_mode(self, mode): calls["aggregate_mode"] = mode

    fake_sdk = SimpleNamespace(
        Context=Context, Pipeline=Pipeline, Config=Config,
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth"),
        OBFormat=SimpleNamespace(RGB="rgb", Y16="y16"),
        OBStreamType=SimpleNamespace(COLOR_STREAM="color"),
        OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        AlignFilter=lambda align_to_stream: object(),
    )
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", fake_sdk)

    stream = camera.CameraStream(
        backend="orbbec", camera_id="FEMTO-001", width=640, height=480, fps=30,
        color_format="RGB",
        depth_mode={"width": 320, "height": 240, "fps": 30.0, "format": "Y16"},
    )
    stream._start_orbbec()

    assert calls["serial"] == "FEMTO-001"
    assert calls["pipeline_device"] is device
    assert calls["requested_profiles"] == [
        ("color", 640, 480, "rgb", 30),
        ("depth", 320, 240, "y16", 30),
    ]
    assert stream.device_identity == "FEMTO-001"
    assert stream.negotiated_mode == {
        "width": 640, "height": 480, "fps": 30.0, "color_format": "RGB",
        "depth_mode": {"width": 320, "height": 240, "fps": 30.0, "format": "Y16"},
        "align_mode": "software_depth_to_color",
        "frame_aggregate_mode": "full_frame_require",
    }
    assert stream.color_intrinsics == {
        "fx": 601.0, "fy": 602.0, "cx": 319.5, "cy": 239.5,
        "width": 640, "height": 480,
    }
    assert stream.color_distortion == {
        "model": "model_api_unexposed",
        "coefficient_order": ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
        "coefficients": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    }
    assert calls["active_color_distortion_queries"] == 1
    assert calls.get("default_profiles", []) == []
    assert calls["aggregate_mode"] == "full"


def test_end_boundary_waits_for_registered_pre_end_orbbec_decode(monkeypatch):
    decode_entered = threading.Event()
    release_decode = threading.Event()
    stream = camera.CameraStream(backend="orbbec")
    color = object()
    depth = SimpleNamespace(get_depth_scale=lambda: 1.0)
    frameset = SimpleNamespace(
        get_color_frame=lambda: color,
        get_depth_frame=lambda: depth,
    )

    class Pipeline:
        calls = 0

        def wait_for_frames(self, _timeout):
            self.calls += 1
            if self.calls == 1:
                return frameset
            stream._stop.wait(0.01)
            return None

    class Align:
        def process(self, value):
            decode_entered.set()
            assert release_decode.wait(1.0)
            return value

    stream._pipe = Pipeline()
    stream._align = Align()
    stream._ob = object()
    monkeypatch.setattr(
        camera, "_ob_color_to_rgb",
        lambda _frame, _sdk: np.zeros((2, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        camera, "_ob_depth_to_array",
        lambda _frame: np.ones((2, 3), dtype=np.uint16),
    )
    monkeypatch.setattr(camera, "_orbbec_device_time_ms", lambda _frame: 123.0)
    stream._stop.clear()
    capture = threading.Thread(target=stream._run_orbbec, daemon=True)
    capture.start()
    assert decode_entered.wait(1.0)

    outcome = {}
    boundary_waiter = threading.Thread(
        target=lambda: outcome.setdefault(
            "boundary", stream.latch_end_boundary(timeout_s=1.0),
        ),
        daemon=True,
    )
    boundary_waiter.start()
    time.sleep(0.02)
    assert boundary_waiter.is_alive(), "End must wait for the registered pre-boundary decode"
    release_decode.set()
    boundary_waiter.join(1.0)
    stream._stop.set()
    capture.join(1.0)

    boundary = outcome["boundary"]
    assert boundary.frame.index == 1
    assert boundary.frame.t <= boundary.t_end
    assert boundary.proof["boundary_latched_atomically"] is True
    assert boundary.proof["inflight_pre_boundary_seen"] is True
    assert boundary.proof["inflight_pre_boundary_completed"] is True
    assert boundary.proof["inflight_pre_boundary_produced_frame"] is True
    assert boundary.proof["snapshot_frame_index"] == 1


def test_end_boundary_snapshot_is_not_replaced_by_post_end_one_slot_publish():
    stream = camera.CameraStream(backend="fake")
    pre = camera.CameraFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        t=time.monotonic() - 0.01,
        index=7,
        status="OK",
    )
    with stream._lock:
        stream._latest = pre
        stream._count = 7

    boundary = stream.latch_end_boundary(timeout_s=0.1)
    post_id, post_t = stream._begin_inflight()
    stream._complete_inflight(
        post_id,
        post_t,
        rgb=np.ones((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
    )

    assert stream.latest_frame().index == 8
    assert boundary.frame.index == 7
    assert stream._end_boundary_snapshot.index == 7
    assert stream.latch_end_boundary(timeout_s=0.1).frame.index == 7
    assert boundary.proof["post_boundary_overwrite_prevented"] is True


def test_end_boundary_timeout_is_fail_closed_while_pre_end_decode_is_inflight():
    stream = camera.CameraStream(backend="orbbec")
    with stream._lock:
        stream._inflight_id = 4
        stream._inflight_host_t = time.monotonic() - 0.01

    with pytest.raises(RuntimeError, match="camera End boundary.*in-flight"):
        stream.latch_end_boundary(timeout_s=0.01)


def test_delivery_fifo_survives_periodic_consumer_stalls_without_overwrite():
    stream = camera.CameraStream(backend="fake", delivery_capacity_frames=8)

    boundary = stream.begin_delivery_interval(timeout_s=0.1)
    assert boundary.proof["start_boundary_frame_index"] == 0

    delivered = []
    for index in range(1, 4):
        inflight_id, host_t = stream._begin_inflight()
        stream._complete_inflight(
            inflight_id,
            host_t,
            rgb=np.full((2, 2, 3), index, dtype=np.uint8),
            depth=np.full((2, 2), index, dtype=np.uint16),
        )
    # Preview may poll the latest slot repeatedly while the lossless consumer is
    # stalled; those reads must neither pop nor overwrite delivery frames.
    assert stream.latest_frame().index == 3
    assert stream.latest_frame().index == 3
    delivered.extend(frame.index for frame in stream.drain_delivery(max_frames=2))

    for index in range(4, 7):
        inflight_id, host_t = stream._begin_inflight()
        stream._complete_inflight(
            inflight_id,
            host_t,
            rgb=np.full((2, 2, 3), index, dtype=np.uint8),
            depth=np.full((2, 2), index, dtype=np.uint16),
        )
    assert stream.latest_frame().index == 6
    delivered.extend(frame.index for frame in stream.drain_delivery())

    assert delivered == [1, 2, 3, 4, 5, 6]
    health = stream.delivery_health()
    assert health["interval_overflow_frames"] == 0
    assert health["interval_enqueued_frames"] == 6
    assert health["interval_delivered_frames"] == 6
    assert health["queue_depth"] == 0


def test_delivery_fifo_overflow_latches_interval_error_and_never_resumes_with_a_gap():
    stream = camera.CameraStream(backend="fake", delivery_capacity_frames=2)
    stream.begin_delivery_interval(timeout_s=0.1)

    for index in range(1, 5):
        inflight_id, host_t = stream._begin_inflight()
        stream._complete_inflight(
            inflight_id,
            host_t,
            rgb=np.full((2, 2, 3), index, dtype=np.uint8),
            depth=np.full((2, 2), index, dtype=np.uint16),
        )

    # The accepted prefix is retained exactly.  Once frame 3 overflows, frame 4
    # must not be appended later and create an apparently contiguous completion.
    assert [frame.index for frame in stream.drain_delivery()] == [1, 2]
    health = stream.delivery_health()
    assert health["status"] == "OVERFLOW"
    assert health["interval_overflow_frames"] == 1
    assert health["first_overflow_frame_index"] == 3
    assert health["interval_enqueued_frames"] == 2
    assert health["interval_delivered_frames"] == 2
    assert health["accepting"] is False


def test_camera_delivery_health_exposes_capture_thread_error_and_frame_freshness(monkeypatch):
    stream = camera.CameraStream(backend="fake", fps=30)
    stream._thread = SimpleNamespace(is_alive=lambda: False)
    stream._thread_error = "RuntimeError: camera exploded"
    stream._latest = camera.CameraFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        t=10.0,
        index=9,
        status="OK",
    )
    monkeypatch.setattr(camera.time, "monotonic", lambda: 10.2)

    health = stream.delivery_health()

    assert health["status"] == "ERROR"
    assert health["thread_alive"] is False
    assert health["thread_error"] == "RuntimeError: camera exploded"
    assert health["last_frame_index"] == 9
    assert health["last_frame_age_s"] == pytest.approx(0.2)
    assert health["freshness_limit_s"] == pytest.approx(0.1)
    assert health["frame_fresh"] is False


def test_end_boundary_rejects_dead_or_too_old_camera_before_sealing(monkeypatch):
    stream = camera.CameraStream(backend="fake", fps=30)
    stream._thread = SimpleNamespace(is_alive=lambda: False)
    stream._latest = camera.CameraFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        t=9.95,
        index=7,
        status="OK",
    )
    monkeypatch.setattr(camera.time, "monotonic", lambda: 10.0)

    with pytest.raises(RuntimeError, match="capture thread is not alive"):
        stream.latch_end_boundary(timeout_s=0.1)

    stream._thread = SimpleNamespace(is_alive=lambda: True)
    stream._latest = camera.CameraFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        t=9.8,
        index=8,
        status="OK",
    )
    with pytest.raises(RuntimeError, match="snapshot is stale"):
        stream.latch_end_boundary(timeout_s=0.1)


def test_recording_frame_divisor_keeps_interval_anchored_phase():
    """Divisor 2: only every 2nd frame after the Start boundary enters the
    delivery FIFO, first delivered is ALWAYS boundary+1 regardless of the
    boundary index parity, and skipped frames never count as overflow."""

    from ausc.perception.camera_stream import CameraStream

    stream = CameraStream(
        backend="fake", width=64, height=48, fps=30,
        recording_frame_divisor=2,
    )
    try:
        stream.start()
        # Wait until some frames have been produced, then arm the interval.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = stream.latest_frame()
            if frame is not None and frame.index >= 3:
                break
            time.sleep(0.01)
        boundary = stream.begin_delivery_interval()
        anchor = boundary.proof["phase_anchor_index"]
        assert anchor == boundary.proof["start_boundary_frame_index"] + 1
        assert boundary.proof["recording_frame_divisor"] == 2

        collected = []
        deadline = time.monotonic() + 5.0
        while len(collected) < 5 and time.monotonic() < deadline:
            collected.extend(stream.drain_delivery())
            time.sleep(0.01)
        assert len(collected) >= 5
        indexes = [frame.index for frame in collected]
        assert indexes[0] == anchor
        assert all(
            later - earlier == 2
            for earlier, later in zip(indexes, indexes[1:])
        )
        health = stream.delivery_health()
        assert int(health.get("interval_overflow_frames", -1)) == 0
    finally:
        stream.stop()
