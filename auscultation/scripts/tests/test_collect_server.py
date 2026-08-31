from __future__ import annotations

import importlib.util
import http.client
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.camera_stream import CameraFrame


SCRIPTS = Path(__file__).resolve().parents[1]
AUSCULTATION = SCRIPTS.parent
for path in (SCRIPTS, AUSCULTATION, AUSCULTATION / "ausc"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mechanical_sensor.force_feedback import force_capture as force_capture_module  # noqa: E402
from mechanical_sensor.force_feedback import force_sensor as force_sensor_module  # noqa: E402
from ausc import realtime_q  # noqa: E402


def _load_collect_server():
    path = SCRIPTS / "collect_server.py"
    spec = importlib.util.spec_from_file_location("collect_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tracking(*, back_present=True, initialized=True, points=None, reason="TRACKING"):
    return {
        "back_present": back_present,
        "initialized": initialized,
        "reason": reason,
        "detector_confidence": 0.91 if back_present else 0.0,
        "points": points or [],
    }


def test_camera_clone_uses_the_canonical_package_frame_type() -> None:
    server = _load_collect_server()
    original = CameraFrame(
        frame=np.zeros((4, 5, 3), dtype=np.uint8),
        depth=np.ones((4, 5), dtype=np.uint16),
        t=1.0,
        index=1,
        status="OK",
        device_t_ms=2.0,
    )

    cloned = server.clone_camera_frame(original)

    assert type(cloned) is CameraFrame


def test_real_preview_never_exports_static_points_when_no_back():
    server = _load_collect_server()
    points = server.visible_tracking_points(
        _tracking(
            back_present=False,
            initialized=False,
            reason="NO_BACK",
            points=[{
                "name": "R1", "pixel": [100.0, 120.0], "status": "TRACKING",
                "measured": True, "position_std": 0.003,
            }],
        ),
    )
    assert points == []


def test_real_preview_waits_for_tracker_initialization():
    server = _load_collect_server()
    points = server.visible_tracking_points(
        _tracking(
            initialized=False,
            reason="INITIALIZING",
            points=[{
                "name": "R1", "pixel": [100.0, 120.0], "status": "TRACKING",
                "measured": True, "position_std": 0.003,
            }],
        ),
    )
    assert points == []


def test_real_preview_uses_model_pixels_and_marks_short_coasts_distinctly():
    server = _load_collect_server()
    result = server.visible_tracking_points(_tracking(points=[
        {
            "name": "R1", "pixel": [101.5, 120.25], "xyz_camera": [0.1, 0.2, 0.9],
            "status": "TRACKING", "measured": True, "position_std": 0.003,
        },
        {
            "name": "L1", "pixel": [210.5, 121.0], "xyz_camera": [-0.1, 0.2, 0.9],
            "status": "COASTING", "measured": False, "position_std": 0.006,
        },
        {
            "name": "R2", "pixel": [111.0, 180.0], "xyz_camera": [0.1, 0.3, 0.9],
            "status": "LOST", "measured": False, "position_std": 0.2,
        },
    ]))
    assert [(point["name"], point["display_status"], point["pixel"]) for point in result] == [
        ("R1", "measured", [101.5, 120.25]),
        ("L1", "coasting", [210.5, 121.0]),
    ]


def test_real_preview_displays_only_authorized_production_occlusion_states():
    server = _load_collect_server()
    result = server.visible_tracking_points(_tracking(points=[
        {
            "name": "R1", "pixel": [101.5, 120.25],
            "status": "DIRECT", "occlusion_state": "DIRECT",
            "valid": True, "measured": True, "robot_occluded": False,
            "source": "current_direct_geometry",
            "direct_geometry_accepted": True, "position_std": 0.003,
        },
        {
            "name": "L1", "pixel": [210.5, 121.0],
            "status": "PATIENT_FRAME_LOCKED",
            "occlusion_state": "PATIENT_FRAME_LOCKED",
            "valid": True, "measured": False, "robot_occluded": True,
            "source": "independent_patient_pose", "position_std": 0.006,
        },
        {
            "name": "R2", "pixel": [111.0, 180.0],
            "status": "PATIENT_FRAME_LOCKED",
            "occlusion_state": "PATIENT_FRAME_LOCKED",
            "valid": True, "measured": False, "robot_occluded": True,
            "source": "landmark_model_coast", "position_std": 0.006,
        },
    ]))

    assert [
        (point["name"], point["display_status"]) for point in result
    ] == [
        ("R1", "measured"),
        ("L1", "patient_locked"),
    ]


def test_dynamic_overlay_draws_nothing_for_no_back():
    server = _load_collect_server()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    before = frame.copy()
    server.draw_tracking_overlay(
        frame,
        _tracking(back_present=False, initialized=False, reason="NO_BACK"),
        active_name="R1",
        completed_names=set(),
    )
    assert np.array_equal(frame, before)


def test_cheap_current_depth_gate_rejects_flat_or_sparse_scenes():
    server = _load_collect_server()
    flat = np.full((1080, 1920), 900, dtype=np.uint16)
    sparse = np.zeros((1080, 1920), dtype=np.uint16)
    sparse[::20, ::20] = 900
    gradient = np.tile(
        np.linspace(650, 1050, 1920, dtype=np.uint16), (1080, 1),
    )

    assert server.depth_back_plausible(flat, 1.0) is False
    assert server.depth_back_plausible(sparse, 1.0) is False
    assert server.depth_back_plausible(gradient, 1.0) is True


def test_real_camera_open_failure_is_fatal_and_never_becomes_fake():
    server = _load_collect_server()

    class BrokenCamera:
        def start(self):
            raise RuntimeError("orbbec busy")

    with pytest.raises(RuntimeError, match="orbbec busy"):
        server.open_camera(
            lambda **_kwargs: BrokenCamera(),
            backend="orbbec",
            index=0,
            camera_id="CL8326300A0",
        )


def test_tracking_payload_keeps_camera_clock_and_model_provenance():
    server = _load_collect_server()

    class Frame:
        index = 7
        t = 123.25
        device_t_ms = 456.0
        age_s = 0.01
        status = "OK"

    payload = server.tracking_payload(
        Frame(),
        _tracking(points=[{
            "name": "R1", "pixel": [50.0, 60.0], "xyz_camera": [0.0, 0.1, 0.8],
            "status": "TRACKING", "measured": True, "position_std": 0.003,
        }]),
        model_release_id="perception_model-68037e6932fd7315af33",
        model_sha256="68037e6932fd7315af33722a69060f0d9838afbc1a60d6f0349da909917f6a14",
    )
    assert payload["frame_index"] == 7
    assert payload["camera_host_t_s"] == 123.25
    assert payload["camera_device_t_ms"] == 456.0
    assert payload["model_release_id"] == "perception_model-68037e6932fd7315af33"
    assert payload["points"][0]["name"] == "R1"


def test_capture_tracking_association_never_pretends_old_inference_is_same_frame():
    server = _load_collect_server()
    frame = server.SimpleNamespace(
        index=10, t=100.2, device_t_ms=2200.0,
    )
    source = {
        "frame_index": 7,
        "camera_host_t_s": 100.0,
        "camera_device_t_ms": 2100.0,
        "back_present": True,
        "initialized": True,
        "reason": "TRACKING",
        "points": [],
    }

    associated = server.associate_tracking_to_camera_frame(frame, source)

    assert associated["frame_index"] == 10
    assert associated["source_tracking_frame_index"] == 7
    assert associated["source_tracking_host_t_s"] == 100.0
    assert associated["source_tracking_device_t_ms"] == 2100.0
    assert associated["tracking_association"] == "carried_forward"
    assert associated["tracking_age_s"] == pytest.approx(0.2)
    assert source["frame_index"] == 7


def test_display_requires_one_complete_unique_ten_point_constellation():
    server = _load_collect_server()
    names = tuple(f"P{index}" for index in range(10))
    points = [
        {
            "name": name, "pixel": [20.0 + index, 30.0 + index],
            "status": "TRACKING", "measured": True, "position_std": 0.003,
        }
        for index, name in enumerate(names)
    ]
    assert len(server.visible_tracking_points(
        _tracking(points=points), expected_names=names,
    )) == 10
    assert server.visible_tracking_points(
        _tracking(points=points[:-1]), expected_names=names,
    ) == []
    assert server.visible_tracking_points(
        _tracking(points=points[:-1] + [dict(points[0])]), expected_names=names,
    ) == []


def test_effective_tracking_hides_stale_or_current_depth_rejected_points():
    server = _load_collect_server()
    names = tuple(f"P{index}" for index in range(10))
    points = [
        {
            "name": name, "pixel": [20.0 + index, 30.0 + index],
            "status": "TRACKING", "measured": True, "position_std": 0.003,
        }
        for index, name in enumerate(names)
    ]
    tracked = {
        **_tracking(points=points),
        "camera_host_t_s": 100.0,
    }
    stale = server.effective_tracking(
        tracked, expected_names=names, now=101.0,
        max_age_s=0.5, current_depth_plausible=True,
    )
    assert stale["reason"] == "TRACKING_STALE"
    assert stale["points"] == []
    assert stale["back_present"] is False

    no_depth = server.effective_tracking(
        tracked, expected_names=names, now=100.1,
        max_age_s=0.5, current_depth_plausible=False,
    )
    assert no_depth["reason"] == "NO_BACK_CURRENT_FRAME"
    assert no_depth["points"] == []


def test_preview_overlay_is_fail_closed_when_tracking_belongs_to_an_older_frame():
    server = _load_collect_server()
    tracking = {
        **_tracking(back_present=True, initialized=True, points=[{
            "name": "R1", "pixel": [10.0, 10.0], "status": "TRACKING",
            "measured": True, "position_std": 0.003,
        }]),
        "frame_index": 4,
        "constellation_valid": True,
    }

    # Bounded lag: a result 1 frame behind the displayed frame stays visible
    # (live 30 fps preview with ~3 fps inference on this box).
    result = server.tracking_for_preview_frame(5, tracking)
    assert result["back_present"] is True
    assert result["constellation_valid"] is True
    assert [point["name"] for point in result["points"]] == ["R1"]

    # Beyond the lag bound the overlay is stripped fail-closed.
    stale = server.tracking_for_preview_frame(5 + 91, tracking)
    assert stale["back_present"] is False
    assert stale["constellation_valid"] is False
    assert stale["reason"] == "MODEL_PENDING_FOR_CURRENT_FRAME"
    assert stale["points"] == []

    # A result claiming a FUTURE frame is never painted.
    future = server.tracking_for_preview_frame(3, tracking)
    assert future["back_present"] is False
    assert future["points"] == []


def test_preview_always_uses_live_camera_frame_never_paired_source(
    tmp_path,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    source_t = time.monotonic()
    source = server.SimpleNamespace(
        index=4,
        t=source_t,
        device_t_ms=400.0,
        age_s=0.0,
        status="OK",
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=np.tile(
            np.linspace(700, 1000, 64, dtype=np.uint16), (48, 1),
        ),
        depth_scale=1.0,
    )
    newer = server.SimpleNamespace(
        index=8,
        t=source_t + 0.12,
        device_t_ms=520.0,
        age_s=0.0,
        status="OK",
        frame=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=np.tile(
            np.linspace(700, 1000, 64, dtype=np.uint16), (48, 1),
        ),
        depth_scale=1.0,
    )
    model_result = {
        "frame_index": 4,
        "back_present": True,
        "initialized": True,
        "reason": "TRACKING",
        "points": [{
            "name": "R1",
            "pixel": [20.0, 20.0],
            "status": "TRACKING",
            "measured": True,
            "position_std": 0.003,
        }],
    }

    class Worker:
        @staticmethod
        def latest():
            return model_result

        health_age = 0.0

        @classmethod
        def health(cls):
            return {
                "status": "OK", "error": None,
                "last_result_age_s": cls.health_age,
            }

    collector.fake = False
    collector._model_worker = Worker()
    collector._frame_ring[4] = source
    try:
        assert collector._consume_model_result() is True

        # Live-frame preview contract (2026-08-11): the preview always shows
        # the newest camera frame regardless of result cadence — the paired
        # source frame is never selected (at ~1 s/frame inference, pairing
        # froze the doctor video and structurally tripped the Start
        # freshness gate). Overlay honesty is enforced downstream by
        # tracking_for_preview_frame's bounded lag + effective_tracking ages.
        selected = collector._select_preview_source(
            newer,
            collector._latest_tracking,
            model_result_updated=True,
        )
        assert selected is not None
        assert selected[0] is newer
        # The retained pairing clone is released, not published.
        assert collector._pending_model_preview is None

        live = collector._select_preview_source(
            newer,
            collector._latest_tracking,
            model_result_updated=False,
        )
        assert live is not None
        assert live[0] is newer
    finally:
        collector._model_worker = None
        collector.shutdown()


def test_perception_stage_does_not_call_gate_only_work_landmark_tracking():
    server = _load_collect_server()

    waiting = server.perception_stage_snapshot(
        model_status="RUNNING",
        tracking={
            "frame_index": 40,
            "reason": "BACK_CONFIRMATION_REQUIRED",
            "initialized": False,
            "posterior_gate": {
                "reason": "ONE_PERSON",
                "person_present": True,
                "operator_confirmed": False,
            },
        },
        back_confirmation={"confirmed": False, "active": False},
        model_release_id="perc_lmh_v9_occ-test",
    )
    assert waiting["person_gate"]["status"] == "RUNNING"
    assert waiting["landmark_model"]["status"] == "LOADED_WAITING_FOR_BACK"
    assert waiting["landmark_model"]["inference_active"] is False
    assert waiting["landmark_model"]["publication_authorized"] is False

    probing = server.perception_stage_snapshot(
        model_status="RUNNING",
        tracking={
            "frame_index": 42,
            "reason": "BACK_CONFIRMATION_REQUIRED",
            "initialized": False,
            "posterior_gate": {
                "reason": "ONE_PERSON",
                "person_present": True,
                "operator_confirmed": False,
            },
            "landmark_probe": {
                "executed": True,
                "frame_index": 42,
                "raw_reason": "INITIALIZING",
                "raw_initialized": False,
                "raw_point_count": 0,
            },
        },
        back_confirmation={"confirmed": False, "active": False},
        model_release_id="perc_lmh_v9_occ-test",
    )
    assert probing["landmark_model"]["status"] == "PROBING"
    assert probing["landmark_model"]["mode"] == "GATED_PROBE"
    assert probing["landmark_model"]["inference_active"] is True
    assert probing["landmark_model"]["publication_authorized"] is False
    assert probing["landmark_model"]["last_result_frame_index"] == 42

    tracking = server.perception_stage_snapshot(
        model_status="RUNNING",
        tracking={
            "frame_index": 44,
            "reason": "TRACKING",
            "initialized": True,
            "posterior_gate": {
                "reason": "OPERATOR_CONFIRMED_BACK",
                "person_present": True,
                "operator_confirmed": True,
            },
        },
        back_confirmation={"confirmed": True, "active": True},
        model_release_id="perc_lmh_v9_occ-test",
    )
    assert tracking["person_gate"]["status"] == "RUNNING"
    assert tracking["landmark_model"]["status"] == "TRACKING"
    assert tracking["landmark_model"]["inference_active"] is True
    assert tracking["landmark_model"]["publication_authorized"] is True
    assert tracking["landmark_model"]["mode"] == "CONFIRMED_TRACKING"
    assert tracking["landmark_model"]["last_result_frame_index"] == 44


def test_ui_reports_person_gate_and_v9_as_separate_stages():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert "s.perception_stages" in page
    assert "Gate '+" in page
    assert "V9 '+" in page


def test_http_end_returns_finalizing_without_waiting_for_slow_verification(tmp_path):
    server = _load_collect_server()
    release = threading.Event()

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def record_event(self, _payload):
            pass

        def write_frame(self, _frame, _tracking):
            pass

        def finalize(self):
            assert release.wait(2.0)
            return {"training_eligible": False}

        def abort(self, _reason):
            release.set()

    platform = server.CollectionPlatform(
        session_root=tmp_path,
        point_names=("R1", "L1"),
        capture_factory=lambda _path: Capture(),
        writer_types={"DiagnosticPerceptionCapture": Capture},
    )
    assert platform.control("start")["ok"] is True

    class Col:
        jpeg = b"jpeg"

        def control(self, action):
            return platform.control(action)

        def state_snapshot(self):
            return platform.state_snapshot()

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2.0)
    try:
        started = time.monotonic()
        body = json.dumps({"action": "end"})
        connection.request(
            "POST", "/control", body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        elapsed = time.monotonic() - started
        assert response.status == 200
        assert elapsed < 0.25
        assert result["state"] == "FINALIZING"
        assert result["accepted"] is True
    finally:
        release.set()
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_http_redo_outside_recording_returns_explicit_409(tmp_path):
    server = _load_collect_server()
    platform = server.CollectionPlatform(
        session_root=tmp_path,
        point_names=("R1",),
        capture_factory=lambda _path: None,
    )

    class Col:
        jpeg = b"jpeg"

        def control(self, action):
            return platform.control(action)

        def state_snapshot(self):
            return platform.state_snapshot()

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2.0)
    try:
        body = json.dumps({"action": "redo"})
        connection.request(
            "POST", "/control", body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 409
        assert result["ok"] is False
        assert result["code"] == "not_recording"
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_http_confirm_back_requires_and_forwards_bound_preview_frame():
    server = _load_collect_server()
    calls = []

    class Col:
        jpeg = b"jpeg"

        def control(
            self, action, *, preview_frame_index=None, preview_frame_token=None,
        ):
            calls.append((action, preview_frame_index, preview_frame_token))
            return {"ok": True, "state": "READY"}

        @staticmethod
        def state_snapshot():
            return {"camera_status": "OK", "model_status": "RUNNING"}

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        for payload, expected_status in (
            ({"action": "confirm_back"}, 400),
            ({"action": "confirm_back", "preview_frame_index": 321}, 400),
            ({
                "action": "confirm_back",
                "preview_frame_index": 321,
                "preview_frame_token": "a" * 64 + "." + "b" * 64,
            }, 200),
        ):
            connection = http.client.HTTPConnection(
                "127.0.0.1", httpd.server_port, timeout=2.0,
            )
            body = json.dumps(payload)
            connection.request(
                "POST", "/control", body,
                {"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            response.read()
            assert response.status == expected_status
            connection.close()
        assert calls == [("confirm_back", 321, "a" * 64 + "." + "b" * 64)]
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_http_frame_jpeg_binds_bytes_to_one_atomic_preview_index():
    server = _load_collect_server()

    class Col:
        jpeg = b"legacy"

        @staticmethod
        def preview_snapshot():
            return {
                "jpeg": b"exact-frame",
                "frame_index": 777,
                "frame_host_t_s": 12.25,
                "generation": 3,
                "confirmable": True,
                "frame_token": "b" * 64 + "." + "c" * 64,
            }

        @staticmethod
        def state_snapshot():
            return {"camera_status": "OK", "model_status": "RUNNING"}

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=2.0,
    )
    try:
        connection.request("GET", "/frame.jpg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("X-Preview-Frame-Index") == "777"
        assert response.getheader("X-Preview-Generation") == "3"
        assert response.getheader("X-Preview-Confirmable") == "1"
        assert response.getheader("X-Preview-Frame-Token") == (
            "b" * 64 + "." + "c" * 64
        )
        assert response.getheader("X-Preview-Frame-Host-T") == "12.25"
        assert response.read() == b"exact-frame"
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_http_placeholder_frame_is_visible_but_has_no_confirmation_token():
    server = _load_collect_server()

    class Col:
        @staticmethod
        def preview_snapshot():
            return {
                "jpeg": b"placeholder",
                "frame_index": 10,
                "frame_host_t_s": None,
                "generation": 4,
                "confirmable": False,
                "frame_token": None,
            }

        @staticmethod
        def state_snapshot():
            return {"camera_status": "RECONNECTING", "model_status": "RECONNECTING"}

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=2.0,
    )
    try:
        connection.request("GET", "/frame.jpg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("X-Preview-Confirmable") == "0"
        assert response.getheader("X-Preview-Frame-Token") is None
        assert response.read() == b"placeholder"
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_mjpeg_client_cannot_block_server_close_and_stream_handles_oserror():
    server = _load_collect_server()

    class Col:
        jpeg = b"jpeg"

        @staticmethod
        def state_snapshot():
            return {"diagnostic_ready": False, "camera_status": "TEST", "model_status": "TEST"}

        @staticmethod
        def control(_action):
            return {"ok": False}

    httpd = server.CollectionHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=2.0,
    )
    response = None
    try:
        connection.request("GET", "/stream")
        response = connection.getresponse()
        assert response.status == 410
        assert b"obsolete" in response.read()
        assert httpd.daemon_threads is True
        assert httpd.block_on_close is False
        httpd.shutdown()
        started = time.monotonic()
        httpd.server_close()
        assert time.monotonic() - started < 0.1
    finally:
        if response is not None:
            response.close()
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=1.0)


def test_http_handler_reuses_http11_connection_with_bounded_404():
    server = _load_collect_server()

    class Col:
        pass

    httpd = server.CollectionHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=2.0,
    )
    try:
        connection.request("GET", "/missing")
        missing = connection.getresponse()
        assert missing.version == 11
        assert missing.getheader("Content-Length") == str(len(missing.read()))
        connection.request("GET", "/")
        page = connection.getresponse()
        assert page.status == 200
        assert page.version == 11
        assert page.read() == b"page"
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=1.0)


def test_frame_endpoint_skips_an_already_rendered_generation():
    server = _load_collect_server()

    class Col:
        @staticmethod
        def preview_snapshot():
            return {
                "jpeg": b"jpeg", "frame_index": 7, "generation": 3,
                "confirmable": False, "frame_token": None,
                "frame_host_t_s": 1.0,
            }

    httpd = server.CollectionHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", httpd.server_port, timeout=2.0,
    )
    try:
        connection.request("GET", "/frame.jpg?after=7&generation=3")
        unchanged = connection.getresponse()
        assert unchanged.status == 204
        assert unchanged.getheader("Content-Length") == "0"
        assert unchanged.read() == b""
        connection.request("GET", "/frame.jpg?after=7&generation=2")
        changed = connection.getresponse()
        assert changed.status == 200
        assert changed.read() == b"jpeg"
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=1.0)


def test_healthz_is_503_when_model_label_is_running_but_result_is_stale():
    server = _load_collect_server()

    class Col:
        jpeg = b"jpeg"

        def state_snapshot(self):
            return {
                "diagnostic_ready": False,
                "camera_status": "OK",
                "model_status": "RUNNING",
            }

        def control(self, _action):
            raise AssertionError("not used")

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2.0)
    try:
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 503
        assert body == {"ok": False, "camera": "OK", "model": "RUNNING"}
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_healthz_stays_ready_when_only_advisory_force_is_unavailable(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    force = {
        "enabled": False,
        "required": True,
        "status": "ERROR",
        "ready": False,
        "recording": False,
        "force_n": None,
        "hard_limit_force_n": 5.5,
        "error": "sensor unavailable",
        "training_eligible": False,
    }
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = lambda **_kwargs: {
        "strict_ok": True,
        "blockers": (),
        "force": force,
        "robot": {"ready": True},
    }
    handler_type = server.make_handler(collector, b"page")
    try:
        state = collector.state_snapshot()
        assert state["force_sensor"]["status"] == "ERROR"
        assert state["diagnostic_ready"] is True

        handler = object.__new__(handler_type)
        handler.path = "/healthz"
        sent = {}

        def capture_send(body, content_type, *, status=200, no_store=False):
            sent.update(
                body=body,
                content_type=content_type,
                status=status,
                no_store=no_store,
            )

        handler._send = capture_send
        handler.do_GET()
        assert sent["status"] == 200
        assert json.loads(sent["body"])["ok"] is True
    finally:
        collector.shutdown()


def test_live_audio_endpoint_streams_bounded_pcm_with_explicit_cursor_headers():
    server = _load_collect_server()

    class Col:
        jpeg = b"jpeg"

        @staticmethod
        def audio_monitor_chunk(cursor, max_samples):
            assert cursor == 12
            assert 1 <= max_samples <= 22_050
            return {
                "samples": np.asarray([-1.0, 0.0, 0.5], dtype=np.float32),
                "sample_rate": 44_100,
                "start_sample": 12,
                "next_sample": 15,
                "overrun": False,
                "status": "OK",
            }

        @staticmethod
        def state_snapshot():
            return {"diagnostic_ready": True, "camera_status": "OK", "model_status": "RUNNING"}

        @staticmethod
        def control(_action):
            raise AssertionError("not used")

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2.0)
    try:
        connection.request("GET", "/audio/pcm?cursor=12&samples=999999")
        response = connection.getresponse()
        raw = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/octet-stream"
        assert response.getheader("X-Audio-Sample-Rate") == "44100"
        assert response.getheader("X-Audio-Start-Sample") == "12"
        assert response.getheader("X-Audio-Next-Sample") == "15"
        assert response.getheader("X-Audio-Overrun") == "0"
        assert np.frombuffer(raw, dtype="<i2").tolist() == [-32767, 0, 16383]
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_cli_accepts_all_pinned_runtime_and_model_identities():
    server = _load_collect_server()
    parser = server.build_parser()
    values = parser.parse_args([
        "--camera-id", "CL8326300A0",
        "--checkpoint", "/models/best.pt",
        "--checkpoint-sha256", "a" * 64,
        "--base-model", "/models/dinov2.pth",
        "--base-model-sha256", "b" * 64,
        "--model-release-id", "perception_model-a",
        "--posterior-gate-checkpoint", "/models/yolo11x-pose.pt",
        "--posterior-gate-checkpoint-sha256", "d" * 64,
        "--posterior-gate-release-id", "posterior_gate-d",
        "--runtime-release-id", "perception_runtime-b",
        "--runtime-lock", "/runtime/lock.json",
        "--runtime-lock-sha256", "c" * 64,
        "--code-release-id", "code_release-c",
        "--audio-device", "Scarlett Solo USB: Audio (hw:2,0)",
        "--audio-sample-rate", "44100",
        "--audio-channels", "2",
        "--audio-pick-channel", "0",
        "--require-audio",
        "--production-blocker", "commissioning missing",
        "--production-activation", "/releases/production_activation-a/activation.json",
        "--production-activation-sha256", "e" * 64,
        "--controller-launch-release",
        "/releases/controller_launch_v1-" + "f" * 64,
        "--controller-launch-trusted-root", "/releases",
        "--controller-launch-contract-sha256", "f" * 64,
    ])
    assert values.camera_id == "CL8326300A0"
    assert values.posterior_gate_checkpoint == "/models/yolo11x-pose.pt"
    assert values.posterior_gate_release_id == "posterior_gate-d"
    assert values.audio_device == "Scarlett Solo USB: Audio (hw:2,0)"
    assert values.audio_sample_rate == 44_100
    assert values.audio_channels == 2
    assert values.audio_pick_channel == 0
    assert values.require_audio is True
    assert values.production_blocker == ["commissioning missing"]
    assert values.production_activation.endswith("/activation.json")
    assert values.production_activation_sha256 == "e" * 64
    assert values.controller_launch_release.endswith("f" * 64)
    assert values.controller_launch_trusted_root == "/releases"
    assert values.controller_launch_contract_sha256 == "f" * 64


def test_dormant_controller_launch_cli_load_is_all_or_none_and_non_executing():
    server = _load_collect_server()
    parser = server.build_parser()
    digest = "a" * 64
    args = parser.parse_args([
        "--controller-launch-release",
        "/immutable/controller_launch_v1-" + digest,
        "--controller-launch-trusted-root", "/immutable",
        "--controller-launch-contract-sha256", digest,
    ])
    pair = object()
    loaded = server.SimpleNamespace(
        release_id="controller_launch_v1-" + digest,
        contract_sha256=digest,
        open_target_status="HIL_PENDING",
    )
    calls = []

    def loader(release, *, trusted_root, expected_contract_sha256, openarm_pair):
        calls.append((release, trusted_root, expected_contract_sha256, openarm_pair))
        return loaded

    result = server._load_dormant_controller_launch(
        args,
        production_runtime=server.SimpleNamespace(openarm_pair=pair),
        loader=loader,
    )

    assert result is loaded
    assert calls == [(
        args.controller_launch_release,
        args.controller_launch_trusted_root,
        digest,
        pair,
    )]

    for missing in (
        "controller_launch_release",
        "controller_launch_trusted_root",
        "controller_launch_contract_sha256",
    ):
        partial = parser.parse_args([])
        for field in (
            "controller_launch_release",
            "controller_launch_trusted_root",
            "controller_launch_contract_sha256",
        ):
            setattr(partial, field, getattr(args, field))
        setattr(partial, missing, None)
        with pytest.raises(ValueError, match="controller launch"):
            server._load_dormant_controller_launch(
                partial,
                production_runtime=server.SimpleNamespace(openarm_pair=pair),
                loader=loader,
            )
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ("fake", "commissioning_preview_only"))
def test_dormant_controller_launch_cli_is_forbidden_outside_real_mode(mode):
    server = _load_collect_server()
    args = server.build_parser().parse_args([
        "--controller-launch-release", "/immutable/controller_launch_v1-" + "b" * 64,
        "--controller-launch-trusted-root", "/immutable",
        "--controller-launch-contract-sha256", "b" * 64,
    ])
    setattr(args, mode, True)
    calls = []

    with pytest.raises(ValueError, match="real production"):
        server._load_dormant_controller_launch(
            args,
            production_runtime=server.SimpleNamespace(openarm_pair=object()),
            loader=lambda *_args, **_kwargs: calls.append("called"),
        )

    assert calls == []


def test_commissioning_preview_cli_is_explicit_and_cannot_use_fake_data():
    server = _load_collect_server()
    parser = server.build_parser()

    values = parser.parse_args(["--commissioning-preview-only"])
    assert values.commissioning_preview_only is True

    with pytest.raises(SystemExit):
        server.main(["--fake", "--commissioning-preview-only"])


def test_commissioning_preview_rejects_every_recording_control_server_side(
    tmp_path,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        camera_backend="orbbec",
        camera_id="CL8326300A0",
        camera_width=1920,
        camera_height=1080,
        camera_fps=30,
        camera_color_format="MJPG",
        depth_width=640,
        depth_height=576,
        depth_fps=30,
        depth_format="Y16",
        out_dir=tmp_path,
        checkpoint="/runtime/best.pt",
        checkpoint_sha256="a" * 64,
        base_model="/runtime/dinov2.pth",
        base_model_sha256="b" * 64,
        model_release_id="perc_lmh_v9_occ-a",
        posterior_gate_checkpoint="/runtime/yolo11x-pose.pt",
        posterior_gate_checkpoint_sha256="c" * 64,
        posterior_gate_release_id="yolo11x_pose-c",
        runtime_release_id="perception_runtime-d",
        runtime_lock="/runtime/lock.json",
        runtime_lock_sha256="d" * 64,
        code_release_id="code_release-e",
        ffmpeg_path="/runtime/ffmpeg",
        ffmpeg_sha256="f" * 64,
        audio_device="Scarlett Solo USB: Audio (hw:2,0)",
        audio_pick_channel=1,
        audio_expected_usb_vid=0x1235,
        audio_expected_usb_pid=0x8211,
        audio_expected_usb_serial="Y77BQVG17CD6E1",
        require_audio=True,
        require_force=False,
        require_robot=False,
        commissioning_preview_only=True,
        start_thread=False,
    )
    try:
        state = collector.state_snapshot()
        assert state["commissioning_preview_only"] is True
        assert state["recording_controls_enabled"] is False
        assert state["capture_mode"] == "COMMISSIONING_PREVIEW_ONLY"

        for action in ("start", "end", "confirm", "redo", "prev", "next"):
            result = collector.control(action)
            assert result["ok"] is False
            assert result["code"] == "commissioning_preview_only"
            assert result["message"] == (
                "Recording is disabled in commissioning preview"
            )
            assert result["state"] == state["state"]
        assert collector.platform.session_dir is None
        assert list(tmp_path.iterdir()) == []
    finally:
        collector.shutdown()


def test_commissioning_preview_main_skips_activation_robot_and_force_but_keeps_audio(
    monkeypatch, tmp_path,
):
    server = _load_collect_server()
    observed = {}

    class FakeCollector:
        def __init__(self, _targets, **kwargs):
            observed.update(kwargs)

        def shutdown(self):
            observed["shutdown"] = True

    class FakeHTTPServer:
        def __init__(self, *_args, **_kwargs):
            pass

        def serve_forever(self):
            return None

        def server_close(self):
            observed["server_closed"] = True

    monkeypatch.setattr(
        server, "load_targets",
        lambda _path: [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
    )
    monkeypatch.setattr(
        server,
        "load_production_runtime_assembly",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preview must not load production activation")
        ),
    )
    monkeypatch.setattr(server, "Collector", FakeCollector)
    monkeypatch.setattr(server, "CollectionHTTPServer", FakeHTTPServer)
    monkeypatch.setattr(
        server, "reconcile_orphan_diagnostic_sessions", lambda *_a, **_k: [],
    )

    server.main([
        "--commissioning-preview-only",
        "--camera-backend", "orbbec",
        "--camera-id", "CL8326300A0",
        "--camera-width", "1920",
        "--camera-height", "1080",
        "--camera-fps", "30",
        "--camera-color-format", "MJPG",
        "--depth-width", "640",
        "--depth-height", "576",
        "--depth-fps", "30",
        "--depth-format", "Y16",
        "--ffmpeg-path", "/runtime/ffmpeg",
        "--ffmpeg-sha256", "f" * 64,
        "--audio-device", "Scarlett Solo USB: Audio (hw:2,0)",
        "--audio-pick-channel", "1",
        "--audio-expected-usb-vid", "0x1235",
        "--audio-expected-usb-pid", "0x8211",
        "--audio-expected-usb-serial", "Y77BQVG17CD6E1",
        "--require-audio",
        "--out", str(tmp_path / "sessions"),
        "--port", "0",
    ])

    assert observed["commissioning_preview_only"] is True
    assert observed["require_robot"] is False
    assert observed["require_force"] is False
    assert observed["robot_capture_coordinator"] is None
    assert observed["audio_device"] == "Scarlett Solo USB: Audio (hw:2,0)"
    assert observed["audio_pick_channel"] == 1
    assert observed["require_audio"] is True
    assert observed["shutdown"] is True
    assert observed["server_closed"] is True


def test_commissioning_preview_injects_completed_handeye_and_unsynced_hall_monitor(
    monkeypatch, tmp_path,
):
    server = _load_collect_server()
    observed = {}
    handeye = object()

    class FakeCollector:
        def __init__(self, _targets, **kwargs):
            observed.update(kwargs)

        def shutdown(self):
            observed["shutdown"] = True

    class FakeHTTPServer:
        def __init__(self, *_args, **_kwargs):
            pass

        def serve_forever(self):
            return None

        def server_close(self):
            pass

    def load_handeye(release, **kwargs):
        observed["handeye_release"] = release
        observed["handeye_loader_kwargs"] = kwargs
        return handeye

    monkeypatch.setattr(
        server, "load_targets",
        lambda _path: [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
    )
    monkeypatch.setattr(server, "load_completed_handeye_release", load_handeye)
    monkeypatch.setattr(server, "Collector", FakeCollector)
    monkeypatch.setattr(server, "CollectionHTTPServer", FakeHTTPServer)
    monkeypatch.setattr(
        server, "reconcile_orphan_diagnostic_sessions", lambda *_a, **_k: [],
    )

    release = tmp_path / ("handeye_activation-" + "a" * 20)
    hall = tmp_path / "hall.json"
    server.main([
        "--commissioning-preview-only",
        "--commissioning-hall-monitor-only",
        "--commissioning-handeye-release", str(release),
        "--commissioning-handeye-artifact-sha256", "a" * 64,
        "--commissioning-handeye-complete-sha256", "b" * 64,
        "--camera-backend", "orbbec",
        "--camera-id", "CL8326300A0",
        "--camera-width", "1920",
        "--camera-height", "1080",
        "--camera-fps", "30",
        "--camera-color-format", "MJPG",
        "--depth-width", "640",
        "--depth-height", "576",
        "--depth-fps", "30",
        "--depth-format", "Y16",
        "--ffmpeg-path", "/runtime/ffmpeg",
        "--ffmpeg-sha256", "f" * 64,
        "--audio-device", "usb-serial:Y77BQVG17CD6E1",
        "--audio-pick-channel", "1",
        "--audio-expected-usb-vid", "0x1235",
        "--audio-expected-usb-pid", "0x8211",
        "--audio-expected-usb-serial", "Y77BQVG17CD6E1",
        "--require-audio",
        "--hall-calibration", str(hall),
        "--hall-calibration-sha256", "c" * 64,
        "--hall-port", "/dev/serial/by-id/hall-53032847188a2e1c",
        "--hall-expected-usb-vid", "0x2e8a",
        "--hall-expected-usb-pid", "0x0005",
        "--hall-expected-usb-serial", "53032847188a2e1c",
        "--out", str(tmp_path / "sessions"),
        "--port", "0",
    ])

    assert observed["handeye_release"] == str(release)
    assert observed["handeye_loader_kwargs"] == {
        "expected_artifact_sha256": "a" * 64,
        "expected_complete_sha256": "b" * 64,
        "expected_camera_id": "CL8326300A0",
    }
    assert observed["handeye_activation"] is handeye
    assert observed["commissioning_hall_monitor_only"] is True
    assert observed["require_force"] is False
    assert observed["require_robot"] is False
    assert observed["shutdown"] is True


def test_commissioning_preview_ui_clearly_disables_recording():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert "Preview only" in page
    assert "recording_controls_enabled" in page
    assert "Recording disabled" in page


def test_real_main_assembles_and_injects_complete_left_runtime(monkeypatch, tmp_path):
    server = _load_collect_server()
    target = server.SimpleNamespace(name="R1", uv=(0.2, 0.2))
    component = object()
    assembly = server.SimpleNamespace(
        handeye=component,
        robot_capture_coordinator=component,
        robot_mask_provider=component,
        back_mask_provider=component,
        patient_pose_provider=component,
        occlusion_runtime=component,
        closed=False,
    )
    assembly.close = lambda: setattr(assembly, "closed", True)
    observed = {}

    def load_activation(path, **kwargs):
        observed["activation_path"] = path
        observed["activation_kwargs"] = kwargs
        return assembly

    class FakeCollector:
        def __init__(self, targets, **kwargs):
            observed["targets"] = targets
            observed["collector_kwargs"] = kwargs

        def shutdown(self):
            observed["collector_shutdown"] = True

    class FakeHTTPServer:
        def __init__(self, *_args, **_kwargs):
            pass

        def serve_forever(self):
            return None

        def server_close(self):
            observed["server_closed"] = True

    monkeypatch.setattr(server, "load_targets", lambda _path: [target])
    monkeypatch.setattr(server, "load_production_runtime_assembly", load_activation)
    monkeypatch.setattr(server, "Collector", FakeCollector)
    monkeypatch.setattr(server, "CollectionHTTPServer", FakeHTTPServer)
    monkeypatch.setattr(server, "reconcile_orphan_diagnostic_sessions", lambda *_a, **_k: [])

    activation = tmp_path / "production_activation-a" / "activation.json"
    server.main([
        "--camera-backend", "orbbec",
        "--camera-id", "CL8326300A0",
        "--camera-width", "1920",
        "--camera-height", "1080",
        "--camera-fps", "30",
        "--camera-color-format", "MJPG",
        "--depth-width", "640",
        "--depth-height", "576",
        "--depth-fps", "30",
        "--depth-format", "Y16",
        "--ffmpeg-path", "/runtime/ffmpeg",
        "--ffmpeg-sha256", "f" * 64,
        "--audio-device", "Scarlett Solo USB: Audio",
        "--audio-pick-channel", "1",
        "--require-audio",
        "--production-activation", str(activation),
        "--production-activation-sha256", "e" * 64,
        "--out", str(tmp_path / "sessions"),
        "--port", "0",
    ])

    assert observed["activation_path"] == str(activation)
    assert observed["activation_kwargs"]["expected_activation_sha256"] == "e" * 64
    assert observed["activation_kwargs"]["point_names"] == ("R1",)
    assert observed["activation_kwargs"]["expected_camera_id"] == "CL8326300A0"
    assert observed["activation_kwargs"]["expected_capture_mode"] == {
        "align_mode": "software_depth_to_color",
        "backend": "orbbec",
        "camera_id": "CL8326300A0",
        "color_format": "MJPG",
        "depth_mode": {
            "format": "Y16", "fps": 30.0, "height": 576, "width": 640,
        },
        "fps": 30.0,
        "frame_aggregate_mode": "full_frame_require",
        "height": 1080,
        "time_source": "orbbec_device_and_host_arrival",
        "width": 1920,
    }
    injected = observed["collector_kwargs"]
    assert injected["require_force"] is True
    assert injected["hall_calibration"] is None
    assert injected["sync_calibration"] is None
    assert injected["require_robot"] is True
    assert injected["handeye_activation"] is component
    assert injected["robot_capture_coordinator"] is component
    assert injected["robot_mask_provider"] is component
    assert injected["back_mask_provider"] is component
    assert injected["patient_pose_provider"] is component
    assert injected["occlusion_runtime"] is component
    assert observed["collector_shutdown"] is True
    assert assembly.closed is True


def test_state_exposes_only_sanitized_provisional_handeye_status(tmp_path):
    server = _load_collect_server()
    handeye = SimpleNamespace(
        handeye_release_id="handeye_activation-" + "a" * 20,
        handeye_sha256="a" * 64,
        calibration_status="USER_ACCEPTED_BELOW_GATE",
        provisional=True,
        quality_gate_passed=False,
        workspace_1sigma_median=0.004956921595,
        workspace_1sigma_max=0.006053641199,
        authorized_scopes=(
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ),
        autonomous_targeting_authorized=False,
        source_frame="orbbec_color_optical_frame",
        target_frame="openarm_left_link0",
        transform_follower_base_from_camera=np.eye(4),
    )
    collector = server.Collector(
        [SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        handeye_activation=handeye,
        start_thread=False,
    )
    try:
        status = collector.state_snapshot()["handeye"]
    finally:
        collector.shutdown()

    assert status == {
        "status": "BOUND_PROVISIONAL",
        "release_id": "handeye_activation-" + "a" * 20,
        "artifact_sha256": "a" * 64,
        "calibration_status": "USER_ACCEPTED_BELOW_GATE",
        "provisional": True,
        "quality_gate_passed": False,
        "workspace_1sigma_median_mm": pytest.approx(4.956921595),
        "workspace_1sigma_max_mm": pytest.approx(6.053641199),
        "authorized_scopes": [
            "bootstrap_teleop_logging",
            "conservative_robot_mask",
        ],
        "autonomous_targeting_authorized": False,
        "source_frame": "orbbec_color_optical_frame",
        "target_frame": "openarm_left_link0",
    }
    assert "transform_follower_base_from_camera" not in status


def test_ui_labels_provisional_handeye_without_claiming_production_ready():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert 'id="handeyestat"' in page
    assert "Hand-eye · Provisional" in page
    assert "autonomous targeting" not in page.lower()


def test_real_collector_requires_an_independent_pinned_posterior_gate(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        camera_id="CL8326300A0",
        camera_width=1920,
        camera_height=1080,
        camera_fps=30,
        camera_color_format="MJPG",
        depth_width=640,
        depth_height=576,
        depth_fps=30,
        depth_format="Y16",
        out_dir=tmp_path,
        checkpoint="/models/v9.pt",
        checkpoint_sha256="a" * 64,
        base_model="/models/dino.pth",
        base_model_sha256="b" * 64,
        model_release_id="v9-a",
        runtime_release_id="runtime-a",
        runtime_lock="/runtime/lock.json",
        runtime_lock_sha256="c" * 64,
        code_release_id="code-a",
        start_thread=False,
    )
    try:
        blockers = collector._configuration_blockers()
        assert "missing posterior-person gate checkpoint path" in blockers
        assert "missing posterior-person gate checkpoint SHA-256" in blockers
        assert "missing posterior-person gate release id" in blockers
    finally:
        collector.shutdown()


def test_capture_video_fps_is_bound_to_negotiated_camera_mode_not_ui_default(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, diagnostic_fps=20.0, start_thread=False,
    )
    collector._provenance["camera"]["negotiated_mode"] = {
        "width": 64, "height": 48, "fps": 30.0,
    }
    session = tmp_path / "fps-session"
    session.mkdir()
    try:
        capture = collector._capture_factory(session)
        assert capture.fps == 30.0
        assert capture.provenance["camera"]["negotiated_mode"]["fps"] == 30.0
        assert capture.require_robot is False
    finally:
        collector.shutdown()


def test_required_left_robot_session_starts_before_camera_is_armed(tmp_path):
    server = _load_collect_server()
    events = []

    class Robot:
        recording = False
        faulted = False
        fault_reason = ""

        def start_session(self, session_dir):
            assert Path(session_dir).is_dir()
            self.recording = True
            events.append("robot_start")

        def authorize_camera_arm(self):
            assert self.recording is True
            events.append("robot_authorize")
            return server.SimpleNamespace(
                pair="left:can1->can3",
                authorized_monotonic_ns=time.monotonic_ns(),
            )

        def abort_session(self, reason):
            events.append(("robot_abort", str(reason)))
            self.recording = False
            return True

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            events.append("capture_start")

        def record_event(self, _payload):
            pass

        def set_recording_boundary(self, payload):
            assert payload["robot_camera_arm"]["pair"] == "left:can1->can3"
            events.append("capture_boundary")

        def abort(self, _reason):
            events.append("capture_abort")

    class Camera:
        @staticmethod
        def begin_delivery_interval(timeout_s=0):
            del timeout_s
            events.append("camera_arm")
            started = time.monotonic()
            return server.SimpleNamespace(
                t_start=started,
                frame=server.SimpleNamespace(
                    index=40, t=started, status="OK",
                    frame=np.zeros((2, 2, 3), dtype=np.uint8),
                ),
                proof={
                    "boundary_latched_atomically": True,
                    "start_boundary_frame_index": 40,
                    "start_boundary_host_t": started,
                },
            )

        @staticmethod
        def abort_delivery_interval():
            pass

        @staticmethod
        def stop():
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector._cam = Camera()
    collector._last_camera_index = 40
    # Production-equivalent Start now requires a controller which has already
    # reached READY; the recording path must never be the arm bring-up path.
    collector._controller_companion = server.SimpleNamespace(
        state="ready", disabled=False,
    )
    preview_episode_id = collector._occlusion_episode_id
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        result = collector.control("start")
        assert result["ok"] is True
        assert collector._occlusion_episode_id.startswith("session-")
        assert collector._occlusion_episode_id != preview_episode_id
        assert events[:5] == [
            "capture_start",
            "robot_start",
            "robot_authorize",
            "camera_arm",
            "capture_boundary",
        ]
    finally:
        collector._abort_recording("test cleanup")
        assert collector._occlusion_episode_id.startswith("preview-")
        collector._controller_companion.disabled = True
        collector.shutdown()


def test_real_collector_fails_configuration_without_left_robot_evidence_chain(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        out_dir=tmp_path,
        require_robot=True,
        start_thread=False,
    )
    try:
        blockers = collector._configuration_blockers()
        assert "missing left OpenArm capture coordinator" in blockers
        assert "missing commissioned OpenArm robot-mask provider" in blockers
        assert "missing independent posterior back-mask provider" in blockers
        assert "missing metric patient-pose provider" in blockers
        assert "missing occlusion safety runtime" in blockers
    finally:
        collector.shutdown()


def test_tracker_processor_receives_complete_occlusion_chain_and_episode_id(
    tmp_path, monkeypatch,
):
    server = _load_collect_server()
    tracking_module = sys.modules["ausc.perception.tracking"]
    calls = {}

    class Resettable:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    back_mask_provider = object()
    patient_pose_provider = Resettable()
    occlusion_runtime = Resettable()

    class Detector:
        def __init__(self, *args, **kwargs):
            calls["detector"] = (args, kwargs)

    class Tracker:
        def __init__(self, *args, **kwargs):
            calls["tracker"] = (args, kwargs)

    class PersonGate:
        pass

    class OperatorGate:
        def __init__(self, person_gate, *, max_observation_age_s, max_preview_lag_frames):
            assert isinstance(person_gate, PersonGate)
            assert max_observation_age_s == 4.0
            assert max_preview_lag_frames == 150

    def tracker_processor(tracker, runtime_frame_fn, **kwargs):
        calls["processor"] = (tracker, runtime_frame_fn, kwargs)
        return "processor"

    monkeypatch.setattr(server, "verify_file_sha256", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module, "NormalsDetector", Detector)
    monkeypatch.setattr(tracking_module, "ModelBasedTracker", Tracker)
    monkeypatch.setattr(tracking_module, "runtime_frame", object())
    monkeypatch.setattr(server, "UltralyticsPersonGate", lambda *_args, **_kwargs: PersonGate())
    monkeypatch.setattr(server, "OperatorConfirmedBackGate", OperatorGate)
    monkeypatch.setattr(server, "TrackerProcessor", tracker_processor)

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=object(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        back_mask_provider=back_mask_provider,
        patient_pose_provider=patient_pose_provider,
        occlusion_runtime=occlusion_runtime,
        start_thread=False,
    )
    collector._cam = server.SimpleNamespace(color_intrinsics={
        "fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0,
        "width": 2, "height": 2,
    })
    frame = server.SimpleNamespace(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        depth_scale=1.0,
    )
    try:
        assert collector._build_tracker_processor(frame) == "processor"
        kwargs = calls["processor"][2]
        assert kwargs["require_robot_mask"] is True
        assert kwargs["back_mask_provider"] is back_mask_provider
        assert kwargs["patient_pose_provider"] is patient_pose_provider
        assert kwargs["occlusion_runtime"] is occlusion_runtime
        assert callable(kwargs["episode_id"])
        assert kwargs["episode_id"]() == collector._occlusion_episode_id
        assert patient_pose_provider.reset_calls == 1
        assert occlusion_runtime.reset_calls == 1
    finally:
        collector.shutdown()


def test_fake_collector_passes_requested_mode_to_the_synthetic_camera(tmp_path):
    server = _load_collect_server()
    calls = []

    class Camera:
        def start(self):
            return self

    def factory(**kwargs):
        calls.append(kwargs)
        return Camera()

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        camera_width=640,
        camera_height=480,
        camera_fps=20,
        camera_factory=factory,
        start_thread=False,
    )
    try:
        collector._open_camera()
        assert calls == [{
            "backend": "fake",
            "index": 0,
            "camera_id": None,
            "rgb": True,
            "delivery_capacity_frames": 32,
            "recording_frame_divisor": 1,
            "width": 640,
            "height": 480,
            "fps": 20,
        }]
    finally:
        collector.shutdown()


def test_collector_allows_redo_when_current_ui_gate_hides_old_model_result(tmp_path):
    server = _load_collect_server()
    targets = [
        server.SimpleNamespace(name=f"P{index}", uv=(0.1, 0.1))
        for index in range(10)
    ]
    collector = server.Collector(
        targets, fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector.platform.back_present = True
        collector.platform.initialized = True
        collector.platform.target_valid = True
        collector._current_tracking = {
            "back_present": False,
            "initialized": False,
            "constellation_valid": False,
            "reason": "TRACKING_STALE",
            "points": [],
        }

        result = collector.control("redo")

        assert result["ok"] is True
        assert result["ack"] == "Redo saved P0 attempt 1; attempt 2 is active"
        assert collector.platform.attempt == 2
    finally:
        collector.platform.abort("test cleanup")
        collector.shutdown()


def test_confirm_fails_closed_when_collector_runtime_thread_is_dead(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        # Start on a healthy runtime...
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._current_tracking = {
            "back_present": True,
            "initialized": True,
            "constellation_valid": True,
            "reason": "TRACKING",
            "points": [{
                "name": "R1", "pixel": [10.0, 10.0], "status": "TRACKING",
                "measured": True, "position_std": 0.003,
            }],
        }
        # ...then the collector runtime thread dies before Confirm.
        collector._runtime_health_snapshot = (
            lambda **_k: {
                "strict_ok": False,
                "blockers": ["collector loop thread is not alive"],
            })

        result = collector.control("confirm")

        assert result["ok"] is False
        assert result["code"] == "runtime_not_ready"
        state = collector.state_snapshot()
        assert state["diagnostic_ready"] is False
        assert state["points"] == []
        assert state["target_valid"] is False
    finally:
        collector._abort_recording("test cleanup")
        collector.shutdown()


@pytest.mark.parametrize("failed_component", ["camera", "model", "preview", "ingress"])
def test_state_clears_points_when_any_live_pipeline_component_is_not_strict_ok(
    tmp_path, failed_component,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    original_preview = collector._preview_worker
    original_ingress = collector._ingress_worker
    collector._thread = SimpleNamespace(is_alive=lambda: True)
    collector._cam = SimpleNamespace()
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._preview_frame_t = time.monotonic()
    collector._current_tracking = {
        "frame_index": 1,
        "back_present": True,
        "initialized": True,
        "constellation_valid": True,
        "reason": "TRACKING",
        "points": [{
            "name": "R1", "pixel": [10.0, 10.0], "status": "TRACKING",
            "measured": True, "position_std": 0.003,
        }],
    }
    if failed_component == "camera":
        collector._camera_status = "ERROR"
    elif failed_component == "model":
        collector._model_worker = SimpleNamespace(health=lambda: {
            "status": "STOPPED", "thread_alive": False, "last_result_age_s": None,
        })
    elif failed_component == "preview":
        collector._preview_worker = SimpleNamespace(health=lambda: {
            "status": "STOPPED", "thread_alive": False,
        })
    else:
        collector._ingress_worker = SimpleNamespace(health=lambda: {
            "status": "STOPPED", "thread_alive": False,
        })
    try:
        state = collector.state_snapshot()
        assert state["diagnostic_ready"] is False
        assert state["points"] == []
        assert state["points_visible"] == 0
        assert state["target_valid"] is False
        assert state["back_present"] is False
    finally:
        collector._model_worker = None
        collector._preview_worker = original_preview
        collector._ingress_worker = original_ingress
        collector._thread = None
        collector._cam = None
        collector.shutdown()


def test_uncaught_collector_loop_exception_is_captured_and_clears_tracking(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._current_tracking = {
        "back_present": True,
        "initialized": True,
        "constellation_valid": True,
        "reason": "TRACKING",
        "points": [{"name": "R1"}],
    }
    collector._loop = lambda: (_ for _ in ()).throw(RuntimeError("loop boom"))

    collector.start()
    collector._thread.join(1.0)

    try:
        assert collector._thread.is_alive() is False
        assert "RuntimeError: loop boom" in collector._loop_error
        state = collector.state_snapshot()
        assert state["diagnostic_ready"] is False
        assert state["points"] == []
        assert state["target_valid"] is False
        assert state["runtime_health"]["collector_loop_error"] == collector._loop_error
    finally:
        collector.shutdown()


def test_page_makes_diagnostic_scope_end_state_and_action_feedback_unambiguous():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert '<html lang="en">' in page
    assert "<h1>Auscultation Capture</h1>" in page
    assert "<strong>CAPTURE MODE</strong><span>Checking authorization" in page
    assert "scopeTitle.textContent='DIAGNOSTIC CAPTURE'" in page
    assert "scopeNote.textContent='Not training data'" in page
    assert "scopeTitle.textContent='PRODUCTION CAPTURE'" in page
    assert "scopeNote.textContent='Raw session · promotion checked after End'" in page
    assert "End &amp; Verify" in page
    assert 'id="actionmsg"' in page
    assert 'id="btnredo"' in page and 'id="btnconfirm"' in page
    assert "object-fit:contain" in page
    assert "aspect-ratio:16/9" in page
    assert "s.target_valid" in page
    assert "s.points_visible===s.n" in page
    assert "s.action_ack!==lastActionAck" in page
    assert "lastRevision" not in page
    assert "const canConfirm=!busy&&recording&&s.target_valid&&s.points_visible===s.n" in page
    assert "const canRedo=!busy&&recording" in page
    assert "$('btnredo').disabled=!canRedo" in page
    assert "let transportDown=false" in page
    assert "const recovered=transportDown;transportDown=false;render(state)" in page
    assert "transportDown=true;action('State disconnected: '" in page
    assert 'id="videoresult"' in page
    assert 'id="storage"' in page
    assert "s.capture_storage" in page
    assert "estimated_remaining_seconds" in page
    assert "usable_bytes_before_safety_reserve" in page
    assert "'Verified · '+proof.n_rgb_frames+' video frames'" in page


def test_state_does_not_publish_a_placeholder_quality_score(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    try:
        state = collector.state_snapshot()
        assert "q" not in state
    finally:
        collector.shutdown()


def test_confirm_back_is_a_distinct_collector_action_not_a_point_confirmation(tmp_path):
    server = _load_collect_server()

    class BackGate:
        def confirm_back(self, *, preview_frame_index):
            assert preview_frame_index == 42
            return {
                "ok": True,
                "code": "back_confirmation_pending",
                "message": "Back confirmed; hold position",
                "back_confirmation": {"active": True, "confirmed": False},
            }

        def confirmation_snapshot(self):
            return {"active": True, "confirmed": False}

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = BackGate()
    collector._preview_frame_confirmable = True
    original = collector.platform.control
    collector.platform.control = lambda action: (_ for _ in ()).throw(
        AssertionError(f"confirm_back leaked to CollectionPlatform: {action}")
    )
    try:
        result = collector.control(
            "confirm_back",
            preview_frame_index=42,
            preview_frame_token=collector._preview_frame_token(42, collector.jpeg),
        )
        assert result["ok"] is True
        assert result["code"] == "back_confirmation_pending"
        assert result["state"] == "READY"
    finally:
        collector.platform.control = original
        collector.shutdown()


def test_confirm_back_rejects_forged_preview_frame_token(tmp_path):
    server = _load_collect_server()

    class BackGate:
        called = False

        @classmethod
        def confirm_back(cls, *, preview_frame_index):
            del preview_frame_index
            cls.called = True
            raise AssertionError("forged preview token reached the back gate")

        @staticmethod
        def confirmation_snapshot():
            return {
                "required": True,
                "active": False,
                "confirmed": False,
                "confirmable": True,
            }

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = BackGate()
    try:
        result = collector.control(
            "confirm_back",
            preview_frame_index=42,
            preview_frame_token="0" * 64,
        )
        assert result["ok"] is False
        assert result["code"] == "preview_frame_not_bound"
        assert BackGate.called is False
    finally:
        collector.shutdown()


def test_preview_token_binds_frame_index_and_exact_jpeg_bytes(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._preview_frame_confirmable = True
        first = collector._preview_frame_token(42, b"jpeg-a")
        second = collector._preview_frame_token(42, b"jpeg-b")

        assert first != second
        assert first.split(".", 1)[0] == hashlib.sha256(b"jpeg-a").hexdigest()
        assert collector._preview_frame_token_valid(42, first) is True
        assert collector._preview_frame_token_valid(43, first) is False
        assert collector._preview_frame_token_valid(42, second) is True
    finally:
        collector.shutdown()


def test_camera_reset_rotates_preview_generation_and_accepts_low_indices(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    with collector._lock:
        collector._preview_result_index = 9000
        collector._preview_frame_index = 9000
        collector.jpeg = b"old-generation-jpeg"
    old_generation = collector._preview_generation
    old_token = collector._preview_frame_token(9000, collector.jpeg)

    try:
        collector._reset_camera_pipeline("camera reconnect test")

        assert collector._preview_generation == old_generation + 1
        assert collector._preview_result_index == -1
        assert collector._preview_frame_index == 0
        assert collector._preview_frame_token_valid(9000, old_token) is False

        stale_frame = SimpleNamespace(index=10, t=time.monotonic())
        collector._publish_preview(
            np.zeros((8, 8, 3), dtype=np.uint8), frame=stale_frame,
        )
        placeholder = collector.preview_snapshot()
        assert placeholder["confirmable"] is False
        assert placeholder["frame_token"] is None
        placeholder_token = collector._preview_frame_token(
            placeholder["frame_index"], placeholder["jpeg"],
        )
        assert collector._preview_frame_token_valid(
            placeholder["frame_index"], placeholder_token,
        ) is False

        replacement = collector._preview_worker
        replacement.stop()
        collector._preview_worker = SimpleNamespace(
            latest=lambda: {
                "frame_index": 1,
                "preview_host_t_s": time.monotonic(),
                "jpeg": b"new-generation-jpeg",
            },
            stop=lambda: None,
        )
        assert collector._consume_preview_result() is True
        assert collector._preview_result_index == 1
        assert collector._preview_frame_index == 1
        assert collector.jpeg == b"new-generation-jpeg"
        assert collector.preview_snapshot()["confirmable"] is True
    finally:
        collector.shutdown()


def test_start_without_active_back_confirmation_creates_nothing(tmp_path):
    server = _load_collect_server()

    class BackGate:
        @staticmethod
        def confirmation_snapshot():
            return {
                "required": True, "mode": "operator", "active": False,
                "confirmed": False, "confirmable": True,
                "reason": "BACK_CONFIRMATION_REQUIRED",
            }

        @staticmethod
        def begin_start_confirmation():
            raise server.BackConfirmationError("Back confirmation is not active")

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = BackGate()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "back_confirmation_required"
        assert collector.platform.session_dir is None
        assert collector._ingress_worker.health()["interval_active"] is False
        assert list(tmp_path.iterdir()) == []
    finally:
        collector.shutdown()


def test_page_separates_back_confirmation_from_point_confirmation():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert 'id="btnback"' in page
    assert "Confirm Back" in page
    assert "Face away, stay alone in frame, then Confirm Back." in page
    assert "ctl('confirm_back')" in page
    assert "displayedPreviewFrameIndex" in page
    assert "displayedPreviewFrameToken" in page
    assert "payload.preview_frame_index=displayedPreviewFrameIndex" in page
    assert "payload.preview_frame_token=displayedPreviewFrameToken" in page
    assert "X-Preview-Confirmable" in page
    assert "displayedPreviewFrameIndex=confirmable?frameIndex:null" in page
    # digest verified when crypto.subtle exists; null (insecure origin, e.g.
    # plain-IP LAN http) skips gracefully instead of killing the video the
    # moment frames turn confirmable at bring-up (2026-08-07)
    assert "digest!==null&&digest!==expectedDigest" in page
    assert "crypto.subtle" in page
    assert "await afterPaint()" in page
    assert "Number(latest?.preview_frame_index)" not in page
    assert 'src="/stream"' not in page
    assert "fetch('/frame.jpg?'+query" in page
    assert "s.back_confirmation" in page
    assert 'id="btnconfirm"' in page
    assert "ctl('confirm')" in page


def test_page_bounds_long_running_video_and_state_resources():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert '<canvas id="camera"' in page
    assert '<img id="camera"' not in page
    assert "createImageBitmap(blob)" in page
    assert "bitmap.close()" in page
    assert "URL.createObjectURL" not in page
    assert "cameraObjectUrl" not in page
    assert "setInterval(poll,250)" not in page
    assert "pollTimer=setTimeout(poll" in page
    assert "document.hidden" in page
    assert "visibilitychange" in page
    assert "pagehide" in page
    assert "source.onended=()=>source.disconnect()" in page
    assert "crypto.subtle.digest('SHA-256',buffer)" in page
    assert "lastBlockersKey" in page
    assert "syncBackButton()" in page
    assert "lastRenderedFrameIndex" in page
    assert "response.status===204" in page


def test_start_revalidates_the_same_confirmation_before_recording_is_armed(tmp_path):
    server = _load_collect_server()

    class BackGate:
        cleared = []

        @staticmethod
        def confirmation_snapshot():
            return {
                "required": True, "active": True, "confirmed": True,
                "confirmable": True, "reason": "OPERATOR_CONFIRMED_BACK",
            }

        @staticmethod
        def begin_start_confirmation():
            return ("claim-1", BackGate.confirmation_snapshot())

        @staticmethod
        def commit_start_confirmation(_token, *, prepare, arm):
            del prepare, arm
            raise server.BackConfirmationError("Back confirmation changed during Start")

        @classmethod
        def clear_confirmation(cls, reason):
            cls.cleared.append(reason)
            return {"required": True, "active": False, "confirmed": False}

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = BackGate()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "back_confirmation_lost_during_start"
        assert collector.platform.state.value == "ABORTED"
        assert collector._capture_ingress_paused is True
        assert collector._ingress_worker.health()["interval_active"] is False
        assert BackGate.cleared
    finally:
        collector.shutdown()


def test_failed_start_aborts_audio_and_clears_back_confirmation(tmp_path):
    server = _load_collect_server()

    class BackGate:
        cleared = []

        @staticmethod
        def confirmation_snapshot():
            return {
                "required": True, "active": True, "confirmed": True,
                "confirmable": True, "reason": "OPERATOR_CONFIRMED_BACK",
            }

        @staticmethod
        def begin_start_confirmation():
            return ("claim-2", BackGate.confirmation_snapshot())

        @staticmethod
        def commit_start_confirmation(_token, *, prepare, arm):
            snapshot = BackGate.confirmation_snapshot()
            prepare(snapshot)
            arm(snapshot)
            return snapshot

        @classmethod
        def clear_confirmation(cls, reason):
            cls.cleared.append(reason)
            return {"required": True, "active": False, "confirmed": False}

    class Audio:
        def __init__(self):
            self.began = False
            self.aborted = False

        def begin_recording(self, _session_dir, *, t0):
            assert t0 > 0
            self.began = True

        def abort_recording(self):
            self.aborted = True

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def set_recording_boundary(self, _payload):
            raise RuntimeError("cannot persist boundary")

        def record_event(self, _payload):
            pass

        def abort(self, _reason):
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    audio = Audio()
    collector._back_gate = BackGate()
    collector._audio = audio
    collector._audio_snapshot = lambda: {"status": "OK"}
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "capture_start_boundary_not_persisted"
        assert audio.began is True
        assert audio.aborted is True
        assert collector._audio_recording_active is False
        assert collector.platform.state.value == "ABORTED"
        assert BackGate.cleared
    finally:
        collector.shutdown()


def test_force_prearms_before_camera_and_uses_exact_shared_boundaries(tmp_path):
    server = _load_collect_server()
    server.validate_capture_boundary_closure = lambda *_args, **_kwargs: {
        "verified": True,
    }
    events = []

    class Force:
        def prepare_recording(self, session_dir):
            assert Path(session_dir).is_dir()
            events.append("force_prearm")

        def begin_recording(self, *, t0):
            events.append(("force_begin", t0))

        def end_recording(self, *, t_end):
            events.append(("force_end", t_end))
            return server.SimpleNamespace(
                path=str(tmp_path / "force.csv"), t_start=123.0, t_end=t_end,
                n_samples=10, status="OK", sha256="a" * 64,
                device_t_start_ms=100.0, device_t_end_ms=200.0,
                calibration_sha256="b" * 64,
                device_identity_sha256="c" * 64,
            )

        def abort_recording(self):
            events.append("force_abort")
            return True

        @staticmethod
        def read():
            return server.SimpleNamespace(
                force_n=2.5, status="WARN", age_s=0.01,
                displacement_mm=1.0, raw=1, hall_v=1.0,
                device_t_ms=100.0, t=123.0,
            )

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            events.append("capture_start")

        def record_event(self, _payload):
            pass

        def set_recording_boundary(self, _payload):
            events.append("capture_boundary")

        def set_force_summary(self, payload):
            assert payload["path"] == "force.csv"
            assert payload["t_start"] == 123.0
            assert payload["t_end"] == 125.0
            events.append("force_bound")

        def set_finalization_context(self, _payload):
            pass

        def finalize(self):
            return {"training_eligible": False}

        def abort(self, _reason):
            events.append("capture_abort")

    class Camera:
        @staticmethod
        def begin_delivery_interval(timeout_s=0):
            del timeout_s
            events.append("camera_start")
            return server.SimpleNamespace(
                t_start=123.0,
                frame=server.SimpleNamespace(index=10, t=123.0),
                proof={
                    "boundary_latched_atomically": True,
                    "start_boundary_frame_index": 10,
                    "start_boundary_host_t": 123.0,
                },
            )

        @staticmethod
        def latch_end_boundary(timeout_s=0, interval_id=None):
            del timeout_s, interval_id
            events.append("camera_end")
            return server.SimpleNamespace(
                t_end=125.0,
                proof={"boundary_latched_atomically": True},
            )

        @staticmethod
        def drain_delivery(max_frames=1):
            del max_frames
            return ()

        @staticmethod
        def delivery_health():
            return {
                "status": "OK", "interval_active": True,
                "queue_depth": 0, "overflow_frames": 0,
                "interval_overflow_frames": 0,
            }

        @staticmethod
        def abort_delivery_interval():
            return None

        @staticmethod
        def stop():
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, require_force=True, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    collector._force = Force()
    collector._cam = Camera()
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector.platform.completion_validator = None
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert events[:5] == [
            "capture_start", "force_prearm", "camera_start",
            ("force_begin", 123.0), "capture_boundary",
        ]
        assert collector.control("end")["ok"] is True
        assert ("force_end", 125.0) in events
        assert events.index(("force_end", 125.0)) > events.index("camera_end")
        assert events.index("force_bound") > events.index(("force_end", 125.0))
    finally:
        collector.shutdown()


@pytest.mark.parametrize("failure_phase", ["prepare", "begin", "end", "bind"])
def test_force_lifecycle_failures_are_advisory_and_end_still_completes(
    tmp_path, failure_phase,
):
    server = _load_collect_server()
    server.validate_capture_boundary_closure = lambda *_args, **_kwargs: {
        "verified": True,
    }
    calls = {"begin": 0, "abort": 0}
    captures = []

    class Force:
        session_dir = None

        def prepare_recording(self, session_dir):
            self.session_dir = Path(session_dir)
            (self.session_dir / "force.csv.inprogress").write_bytes(
                b"host_t_s,force_n,status\n123.0,1.5,OK\n"
            )
            (self.session_dir / "force_capture.json.inprogress").write_text(
                '{"status":"recording"}\n'
            )
            if failure_phase == "prepare":
                raise RuntimeError("prepare failed")

        def begin_recording(self, *, t0):
            calls["begin"] += 1
            if failure_phase == "prepare":
                raise AssertionError("begin called after failed prepare")
            if failure_phase == "begin":
                raise RuntimeError("begin failed")
            assert t0 == 123.0

        def end_recording(self, *, t_end):
            assert self.session_dir is not None
            if failure_phase == "end":
                raise RuntimeError("end failed")
            csv_path = self.session_dir / "force.csv"
            metadata_path = self.session_dir / "force_capture.json"
            (self.session_dir / "force.csv.inprogress").replace(csv_path)
            (self.session_dir / "force_capture.json.inprogress").replace(
                metadata_path
            )
            (self.session_dir / "FORCE_CAPTURE_COMPLETE.json").write_text(
                '{"status":"complete"}\n'
            )
            (self.session_dir / "force_calibration.json").write_text(
                '{"calibrated":true}\n'
            )
            return server.SimpleNamespace(
                path=str(csv_path),
                t_start=123.0,
                t_end=t_end,
                n_samples=1,
                status="OK",
                sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                device_t_start_ms=100.0,
                device_t_end_ms=110.0,
                calibration_sha256="b" * 64,
                device_identity_sha256="c" * 64,
            )

        def abort_recording(self):
            calls["abort"] += 1
            return True

        @staticmethod
        def read():
            return server.SimpleNamespace(
                force_n=1.5,
                status="OK",
                age_s=0.01,
                displacement_mm=0.5,
                raw=1,
                hall_v=1.0,
                device_t_ms=100.0,
                t=123.0,
            )

    class Capture:
        def __init__(self):
            self.recording_boundary = None
            self.finalization_context = None
            captures.append(self)

        def request_cancel(self):
            pass

        def start(self):
            pass

        def record_event(self, _payload):
            pass

        def set_recording_boundary(self, payload):
            self.recording_boundary = json.loads(json.dumps(payload))

        def set_force_summary(self, _payload):
            if failure_phase == "bind":
                raise RuntimeError("bind failed")

        def set_finalization_context(self, payload):
            self.finalization_context = json.loads(json.dumps(payload))

        def finalize(self):
            session_dir = Path(collector.platform.session_dir)
            forbidden = {
                "force.csv",
                "force_capture.json",
                "FORCE_CAPTURE_COMPLETE.json",
                "force_calibration.json",
                "force.csv.inprogress",
                "force_capture.json.inprogress",
            }
            assert not any((session_dir / name).exists() for name in forbidden)
            return {
                "status": "diagnostic_complete",
                "training_eligible": False,
                "contact_force_valid": False,
                "force": None,
                "force_authenticity": None,
                "capture_boundary": self.finalization_context,
            }

        def abort(self, _reason):
            pass

    class Camera:
        active = False

        @classmethod
        def begin_delivery_interval(cls, timeout_s=0):
            del timeout_s
            cls.active = True
            return server.SimpleNamespace(
                t_start=123.0,
                frame=server.SimpleNamespace(index=10, t=123.0),
                interval_id=1,
                proof={
                    "boundary_latched_atomically": True,
                    "start_boundary_frame_index": 10,
                    "start_boundary_host_t": 123.0,
                    "interval_id": 1,
                },
            )

        @classmethod
        def latch_end_boundary(cls, timeout_s=0, interval_id=None):
            del timeout_s
            assert interval_id == 1
            cls.active = False
            return server.SimpleNamespace(
                t_end=125.0,
                proof={
                    "boundary_latched_atomically": True,
                    "post_boundary_overwrite_prevented": True,
                },
            )

        @staticmethod
        def drain_delivery(max_frames=1):
            del max_frames
            return ()

        @classmethod
        def delivery_health(cls):
            return {
                "status": "OK",
                "interval_active": cls.active,
                "accepting": cls.active,
                "queue_depth": 0,
                "overflow_frames": 0,
                "interval_overflow_frames": 0,
                "interval_enqueued_frames": 0,
                "interval_delivered_frames": 0,
            }

        @staticmethod
        def abort_delivery_interval():
            Camera.active = False

        @staticmethod
        def stop():
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    collector._force = Force()
    collector._cam = Camera()
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector.platform.completion_validator = None
    collector._storage_start_rejection = lambda _duration: None
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = lambda **_kwargs: {
        "strict_ok": True,
        "blockers": [],
    }
    try:
        assert collector.control("start")["ok"] is True
        assert collector.control("end")["ok"] is True
        terminal = collector.platform.wait_for_terminal(timeout=2.0)
        assert terminal["state"] == "COMPLETE", terminal.get("error")
        assert calls["abort"] >= 1
        if failure_phase == "prepare":
            assert calls["begin"] == 0
        boundary = captures[0].finalization_context
        assert boundary["force_advisory"]
        event = boundary["force_advisory"][-1]
        assert event["status"] == "DEGRADED"
        assert event["contact_force_valid"] is False
        preserved = event["preserved_artifacts"]
        assert preserved
        for artifact in preserved:
            path = Path(collector.platform.session_dir) / artifact["path"]
            assert path.is_file()
            assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert artifact["bytes"] == path.stat().st_size
    finally:
        collector.shutdown()


def test_force_unsafe_event_is_advisory_and_never_aborts_recording(tmp_path):
    # P0-4: the Hall force reading is an advisory reference for physicians, NOT a
    # safety interlock. An over-threshold reading updates the displayed status only;
    # it must NEVER abort recording (contact-force safety is mechanical + human teleop
    # + the controller E-stop path, not this reading).
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, require_force=True, start_thread=False,
    )

    class Force:
        aborted = False

        @staticmethod
        def prepare_recording(_session_dir):
            pass

        @staticmethod
        def begin_recording(*, t0):
            assert t0 > 0

        @staticmethod
        def read():
            return server.SimpleNamespace(
                force_n=3.25, status="WARN", age_s=0.012,
                displacement_mm=1.0, raw=1, hall_v=1.0,
                device_t_ms=100.0, t=time.monotonic(),
            )

        def abort_recording(self):
            self.aborted = True
            return True

        @staticmethod
        def stop():
            pass

    force = Force()
    collector._force = force
    collector._force_status = "RUNNING"
    try:
        state = collector.state_snapshot()
        assert state["force"] == 3.25
        assert state["force_sensor"]["status"] == "WARN"
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._on_force_safety_event(server.SimpleNamespace(
            force_n=9.5, status="SCREENED_LIMIT", age_s=0.0,
        ))
        # advisory: recording keeps going and the force stream is NOT aborted
        assert collector.platform.state.value == "RECORDING"
        assert force.aborted is False
        # the over-threshold value is surfaced for the physician's reference only
        assert collector._force_status == "SCREENED_LIMIT"
    finally:
        collector.shutdown()


def test_force_capture_fault_is_advisory_and_never_aborts_recording(tmp_path):
    # P0-4: a Hall sensor dropout degrades the advisory force channel but never aborts
    # the session or touches the arm.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, require_force=True, start_thread=False,
    )

    class Force:
        aborted = False

        @staticmethod
        def prepare_recording(_session_dir):
            pass

        @staticmethod
        def begin_recording(*, t0):
            assert t0 > 0

        @staticmethod
        def read():
            return server.SimpleNamespace(
                force_n=1.5, status="OK", age_s=0.01,
                displacement_mm=0.5, raw=1, hall_v=1.0,
                device_t_ms=50.0, t=time.monotonic(),
            )

        def abort_recording(self):
            self.aborted = True
            return True

        @staticmethod
        def stop():
            pass

    force = Force()
    collector._force = force
    collector._force_status = "RUNNING"
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._on_force_capture_fault(RuntimeError("serial dropout"))
        assert collector.platform.state.value == "RECORDING"   # keeps recording
        assert force.aborted is False
        assert collector._force_status == "ERROR"              # channel marked degraded
    finally:
        collector.shutdown()


def test_page_renders_contact_force_as_status_and_progress_without_a_number():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert 'id="forcestat"' not in page
    assert 'id="forcebar"' in page
    assert 'id="forcefill"' in page
    assert 'role="progressbar"' in page
    assert "s.force_sensor" in page
    assert ".toFixed(2)+' N'" not in page
    assert "fs.hard_limit_force_n" in page
    assert "fs.force_n" in page
    assert "fs.status==='WARN'?'var(--amber)'" in page


def test_page_uses_compact_three_card_rail_and_moves_capture_below_video():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    capture = page.index('id="capturestrip"')
    system_details = page.index("<summary>System details</summary>")
    rail = page.index('<aside class="rail">')
    blockers = page.index('id="blockers"')

    assert capture < system_details < blockers < rail
    assert page.count('class="card railcard"') == 3
    assert page.count('id="taskcard"') == 1
    assert page.count('id="sensorcard"') == 1
    assert page.count('id="controlcard"') == 1
    assert page.index('id="btnback"') < page.index('id="plist"')
    assert page.index('id="plist"') < page.index('id="sensorcard"')
    assert ">Before training<" not in page


def test_page_labels_raw_production_capture_without_claiming_promotion(tmp_path):
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")
    assert '<div class="scope"><strong>CAPTURE MODE</strong>' in page
    assert "PRODUCTION CAPTURE" in page
    assert "Raw session · promotion checked after End" in page
    assert "Raw session · training-ready" not in page
    assert '<span class="label blockerhead">Start gate</span>' in page
    assert "gateReasons.unshift(s.start_rejection)" in page

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    try:
        collector.production_teleop = True
        assert collector.state_snapshot()["capture_mode"] == "PRODUCTION_TELEOP_V4"
    finally:
        collector.shutdown()


def test_disabled_force_snapshot_never_claims_zero_or_ready(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        force = collector._force_snapshot()
        assert force["enabled"] is False
        assert force["ready"] is False
        assert force["force_n"] is None
        assert force["training_eligible"] is False
    finally:
        collector.shutdown()


def test_advisory_force_configuration_never_enters_start_blockers(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        out_dir=tmp_path,
        require_force=True,
        require_audio=False,
        start_thread=False,
    )
    try:
        blockers = collector._configuration_blockers()
        assert not any("Hall-force" in blocker for blocker in blockers)
        assert not any("force calibration" in blocker for blocker in blockers)
    finally:
        collector.shutdown()


def test_force_initialization_failure_is_visible_but_never_blocks_capture(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    try:
        blocker_updates = []
        original_set_blockers = collector._set_diagnostic_blockers

        def observe_blockers(blockers):
            blocker_updates.append(tuple(blockers))
            return original_set_blockers(blockers)

        collector._set_diagnostic_blockers = observe_blockers
        collector.start()
        state = collector.platform.state_snapshot()
        force = collector._provenance["force"]
        assert collector._force_status == "ERROR"
        assert not any(
            "force" in blocker.lower()
            for update in blocker_updates
            for blocker in update
        )
        assert not any(
            "force" in blocker.lower()
            for blocker in state["diagnostic_blockers"]
        )
        assert force["advisory"] is True
        assert force["available"] is False
        assert force["status"] == "ERROR"
        assert isinstance(force["error"], str) and force["error"]
    finally:
        collector.shutdown()


def test_force_config_rejects_symlink_and_dynamic_pointer_before_hash(tmp_path):
    server = _load_collect_server()
    source = tmp_path / "calibration.json"
    source.write_text("{}\n")
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    dynamic_dir = tmp_path / "current"
    dynamic_dir.mkdir()
    dynamic = dynamic_dir / "calibration.json"
    dynamic.write_text("{}\n")

    with pytest.raises(RuntimeError, match="non-symlink"):
        server._verified_immutable_input(alias, label="Hall-force calibration")
    with pytest.raises(RuntimeError, match="dynamic pointer"):
        server._verified_immutable_input(dynamic, label="audio-force sync calibration")


def test_force_sync_mode_uses_the_already_opened_audio_and_hall_identities(tmp_path):
    server = _load_collect_server()
    hall_calibration = tmp_path / "hall-calibration.json"
    sync_calibration = tmp_path / "sync-calibration.json"
    hall_calibration.write_text(json.dumps({
        "calibrated": True,
        "target_force_n": 4.0,
        "target_tolerance_n": 0.5,
        "max_screened_force_n": 4.6,
        "hard_limit_force_n": 5.5,
        "calibration_points": [
            {"hall_v": 0.5, "force_n": 0.0, "displacement_mm": 0.0},
            {"hall_v": 1.0, "force_n": 6.0, "displacement_mm": 6.0},
        ],
    }) + "\n")
    sync_calibration.write_text("{}\n")
    hall_sha = hashlib.sha256(hall_calibration.read_bytes()).hexdigest()
    sync_sha = hashlib.sha256(sync_calibration.read_bytes()).hexdigest()
    audio_identity = {
        "device_name": "Scarlett Solo USB: Audio (hw:2,0)",
        "hostapi_name": "ALSA",
        "portaudio_index": 7,
        "max_input_channels": 2,
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_serial": "Y77BQVG17CD6E1",
        "usb_port_path": None,
    }
    hall_identity = {
        "requested_port": (
            "/dev/serial/by-id/"
            "usb-MicroPython_Board_in_FS_mode_53032847188a2e1c-if00"
        ),
        "device": "/dev/ttyACM0",
        "usb_vid": 0x2E8A,
        "usb_pid": 0x0005,
        "usb_serial": "53032847188a2e1c",
        "manufacturer": "Raspberry Pi",
        "product": "RP2040",
        "location": "1-3",
    }
    observed = {}

    class Audio:
        def start(self):
            return self

        def stop(self):
            pass

    class Force:
        def prepare(self):
            return self

        def stop(self):
            pass

        @staticmethod
        def read():
            return server.SimpleNamespace(
                force_n=2.0, status="OK", age_s=0.01,
                displacement_mm=1.0, raw=1, hall_v=1.0,
                device_t_ms=1.0,
            )

    def mode_builder(**kwargs):
        observed["hall"] = kwargs["hall_resolver"]("changed-port")
        observed["audio"] = kwargs["audio_resolver"]("changed-device")
        return {"bound": True}

    sync_metadata = {
        "calibration_sha256": sync_sha,
        "offset_definition": "force_t_minus_audio_t",
        "audio_to_force_offset_ms": 1.25,
        "offset_std_ms": 0.2,
        "n_matched": 8,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "runtime_capture_mode": {"bound": True},
        "identity_match": {"hall_usb": "matched", "audio_usb": "matched"},
    }
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        audio_device="Scarlett Solo USB: Audio (hw:2,0)",
        audio_sample_rate=48_000,
        audio_channels=2,
        audio_pick_channel=1,
        audio_expected_usb_vid=0x1235,
        audio_expected_usb_pid=0x8211,
        audio_expected_usb_serial="Y77BQVG17CD6E1",
        audio_expected_usb_port_path=None,
        require_audio=True,
        hall_calibration=str(hall_calibration),
        hall_calibration_sha256=hall_sha,
        hall_port=(
            "/dev/serial/by-id/"
            "usb-MicroPython_Board_in_FS_mode_53032847188a2e1c-if00"
        ),
        hall_expected_usb_vid=0x2E8A,
        hall_expected_usb_pid=0x0005,
        hall_expected_usb_serial="53032847188a2e1c",
        sync_calibration=str(sync_calibration),
        sync_calibration_sha256=sync_sha,
        require_force=True,
        audio_factory=lambda **_kwargs: Audio(),
        audio_resolver=lambda _device: dict(audio_identity),
        hall_resolver=lambda _port: dict(hall_identity),
        sync_mode_builder=mode_builder,
        sync_loader=lambda _path, **_kwargs: dict(sync_metadata),
        force_factory=lambda *_args, **_kwargs: Force(),
        start_thread=False,
    )
    try:
        collector.start()
        assert observed == {"hall": hall_identity, "audio": audio_identity}
        assert collector._provenance["force"]["sync"] == sync_metadata
        assert collector._provenance["force"]["hardware"] == hall_identity
    finally:
        collector.shutdown()


def test_runtime_abort_continues_after_audio_cleanup_oserror(tmp_path):
    server = _load_collect_server()

    class Audio:
        @staticmethod
        def abort_recording():
            raise OSError("audio writer disappeared")

        @staticmethod
        def stop():
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._audio = Audio()
        collector._audio_recording_active = True

        assert collector._abort_recording("runtime fault") is True
        assert collector._audio_recording_active is False
        assert collector.platform.state.value == "ABORTED"
    finally:
        collector.shutdown()


def test_shutdown_continues_after_audio_stop_oserror(tmp_path):
    server = _load_collect_server()

    class Audio:
        @staticmethod
        def stop():
            raise OSError("audio device vanished")

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    assert collector.control("start")["ok"] is True
    collector._audio = Audio()
    collector._audio_recording_active = True

    collector.shutdown()

    assert collector._audio_recording_active is False
    assert collector.platform.state.value == "ABORTED"


def test_boundary_persistence_cannot_clear_gate_then_arm_recording(tmp_path):
    server = _load_collect_server()
    now = [80.0]
    person = {
        "person_present": True,
        "reason": "ONE_PERSON",
        "box": [100.0, 80.0, 800.0, 980.0],
        "box_confidence": 0.95,
        "box_area_fraction": 0.63,
        "detected_people": 1,
        "scene_people": 1,
        "qualified_people": 1,
        "invalid_detections": 0,
        "frame_shape": [1000, 1000],
    }
    raw_gate = SimpleNamespace(assess=lambda frame: {
        **person, "frame_index": frame.index,
    })
    gate = server.OperatorConfirmedBackGate(raw_gate, clock=lambda: now[0])
    gate.assess(SimpleNamespace(index=1, t=80.0))
    assert gate.confirm_back(preview_frame_index=1)["ok"] is True
    gate.assess(SimpleNamespace(index=2, t=80.01))
    assert gate.confirmation_snapshot()["confirmed"] is True

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def set_recording_boundary(self, _payload):
            gate.clear_confirmation("TEST_SETTER_INVALIDATION")

        def record_event(self, _payload):
            pass

        def abort(self, _reason):
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = gate
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "back_confirmation_lost_during_start"
        assert collector.platform.state.value == "ABORTED"
        assert collector._capture_ingress_paused is True
    finally:
        collector.shutdown()


def test_page_prioritizes_doctor_readable_fps_session_frames_and_live_audio():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert 'id="livefps"' in page
    assert 'id="sessionframes"' in page
    assert 'id="audiostatus"' in page
    assert 'id="btnaudio"' in page
    assert page.count("Start when ready; end to save and verify.") == 1
    assert "Live FPS" in page
    assert "Session frames" in page
    assert 'id="preview_frame_index"' not in page
    assert "MODEL/CAPTURE DROP" not in page


def test_page_labels_robot_send_evidence_as_tx_enqueue_only():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert "TX enqueue only" in page
    assert "SocketCAN TX queued" not in page
    assert "motor acknowledged" not in page.lower()
    assert "motor ack" not in page.lower()


def test_page_uses_short_english_controls_and_blocker_labels():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    for text in (
        "Start", "End &amp; Verify", "Confirm", "Redo", "Previous", "Next",
        "Start gate", "Camera latency", "Motor stop",
        "DISABLED:'Off'", "WAITING:'Waiting'", "No signal",
    ):
        assert text in page
    assert re.search(r"[\u4e00-\u9fff]", page) is None
    for verbose in (
        "Start diagnostic recording", "Recording controls", "Session lifecycle",
        "RGB-D、模型和肺音会保存并校验",
    ):
        assert verbose not in page


def test_live_audio_player_stops_instead_of_replaying_stale_ring_samples():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    assert "X-Audio-Status" in page
    assert "audioStatus!=='OK'" in page
    assert "audioStatus.toLowerCase()" in page


def test_audio_snapshot_distinguishes_open_stream_from_missing_physical_signal(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.audio_device = "hw:3,0"
    collector._audio = server.SimpleNamespace(
        level_db=lambda: server.SimpleNamespace(
            status="OK", rms_db=-96.0, peak_db=-84.0,
        ),
    )

    state = collector._audio_snapshot()

    assert state["status"] == "OK"
    assert state["signal_status"] == "SILENT"
    assert state["signal_present"] is False


def test_audio_snapshot_marks_clear_physical_signal_present(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.audio_device = "hw:3,0"
    collector._audio = server.SimpleNamespace(
        level_db=lambda: server.SimpleNamespace(
            status="OK", rms_db=-31.0, peak_db=-18.0,
        ),
    )

    state = collector._audio_snapshot()

    assert state["signal_status"] == "PRESENT"
    assert state["signal_present"] is True


def test_audio_identity_accepts_same_usb_serial_after_portaudio_index_remap():
    server = _load_collect_server()
    observed = {
        "requested_device": "Scarlett Solo USB: Audio",
        "portaudio_index": 17,
        "hostapi_name": "ALSA",
        "device_name": "Scarlett Solo USB: Audio (hw:7,0)",
        "max_input_channels": 2,
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_serial": "Y77BQVG17CD6E1",
        "usb_port_path": "9-2.2",
        "usb_identity_source": "linux_alsa_sysfs",
    }

    validated = server.validate_audio_hardware_descriptor(
        observed,
        expected_vid=0x1235,
        expected_pid=0x8211,
        expected_serial="Y77BQVG17CD6E1",
        expected_port_path=None,
        channels=2,
        pick_channel=1,
    )

    assert validated["portaudio_index"] == 17
    assert validated["usb_serial"] == "Y77BQVG17CD6E1"


def test_audio_identity_rejects_wrong_serial_and_moved_serialless_topology():
    server = _load_collect_server()
    observed = {
        "requested_device": "Scarlett Solo USB: Audio",
        "portaudio_index": 8,
        "hostapi_name": "ALSA",
        "device_name": "Scarlett Solo USB: Audio (hw:2,0)",
        "max_input_channels": 2,
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_serial": "WRONG",
        "usb_port_path": "9-2.2",
        "usb_identity_source": "linux_alsa_sysfs",
    }
    with pytest.raises(RuntimeError, match="USB serial"):
        server.validate_audio_hardware_descriptor(
            observed,
            expected_vid=0x1235,
            expected_pid=0x8211,
            expected_serial="Y77BQVG17CD6E1",
            expected_port_path=None,
            channels=2,
            pick_channel=1,
        )

    serialless = {**observed, "usb_serial": None, "usb_port_path": "1-13"}
    with pytest.raises(RuntimeError, match="USB topology"):
        server.validate_audio_hardware_descriptor(
            serialless,
            expected_vid=0x1235,
            expected_pid=0x8211,
            expected_serial=None,
            expected_port_path="1-12",
            channels=2,
            pick_channel=1,
        )


def test_collector_opens_resolved_audio_index_and_persists_observed_identity(tmp_path):
    server = _load_collect_server()
    observed = {
        "requested_device": "Scarlett Solo USB: Audio",
        "portaudio_index": 17,
        "hostapi_name": "ALSA",
        "device_name": "Scarlett Solo USB: Audio (hw:7,0)",
        "max_input_channels": 2,
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_serial": "Y77BQVG17CD6E1",
        "usb_port_path": "9-2.2",
        "usb_identity_source": "linux_alsa_sysfs",
    }

    class Audio:
        opened = None

        def __init__(self, **kwargs):
            Audio.opened = dict(kwargs)

        def start(self):
            return self

        def stop(self):
            return None

        def level_db(self):
            return server.SimpleNamespace(status="OK", rms_db=-22.0, peak_db=-13.0)

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        audio_device="Scarlett Solo USB: Audio",
        audio_sample_rate=48_000,
        audio_channels=2,
        audio_pick_channel=1,
        require_audio=True,
        audio_expected_usb_vid=0x1235,
        audio_expected_usb_pid=0x8211,
        audio_expected_usb_serial="Y77BQVG17CD6E1",
        audio_resolver=lambda _device: observed,
        audio_factory=Audio,
        start_thread=False,
    )
    try:
        collector.start()
        state = collector.state_snapshot()

        assert Audio.opened["device"] == 17
        assert Audio.opened["sr_in"] == 48_000
        assert Audio.opened["channels"] == 2
        assert Audio.opened["pick_channel"] == 1
        assert state["audio"]["hardware"]["usb_serial"] == "Y77BQVG17CD6E1"
        assert collector._provenance["audio"]["hardware"] == observed
    finally:
        collector.shutdown()


def test_start_fails_closed_when_audio_device_is_open_but_signal_is_silent(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.audio_device = "Scarlett Solo USB: Audio"
    collector.require_audio = True
    collector._audio_status = "RUNNING"
    collector._audio = server.SimpleNamespace(
        level_db=lambda: server.SimpleNamespace(
            status="OK", rms_db=-96.0, peak_db=-84.0,
        ),
    )
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    collector._runtime_health_snapshot = lambda now=None: {
        "strict_ok": True, "blockers": [],
    }
    try:
        assert collector.state_snapshot()["diagnostic_ready"] is False
        result = collector.control("start")

        assert result["ok"] is False
        assert result["code"] == "audio_signal_missing"
        assert collector.platform.session_dir is None
        assert list(tmp_path.iterdir()) == []
    finally:
        collector.shutdown()


def test_idle_state_reports_storage_capacity_before_recording(tmp_path, monkeypatch):
    server = _load_collect_server()
    free = 100 * 1024 ** 3
    monkeypatch.setattr(
        server.shutil,
        "disk_usage",
        lambda _path: server.shutil._ntuple_diskusage(total=200 * 1024 ** 3, used=100 * 1024 ** 3, free=free),
    )
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        camera_width=1920,
        camera_height=1080,
        depth_width=640,
        depth_height=576,
        depth_fps=30,
        diagnostic_min_free_bytes=50 * 1024 ** 3,
        start_thread=False,
    )

    storage = collector.state_snapshot()["capture_storage"]

    # Diagnostic capture persists depth aligned to the RGB frame (camera size),
    # not the raw depth-mode size, so the raw base uses 1920x1080. The recordable
    # estimate credits the lossless zlib depth codec (measured >=17x on real
    # frames; conservative default ratio 8.0) plus the non-depth allowance and
    # rate margin -- the SAME formula as the Start storage gate, so the idle
    # minutes and the gate can never disagree.
    raw_depth_rate = 1920 * 1080 * 2 * 30
    expected_rate = (raw_depth_rate / 8.0 + 12_000_000) * 1.3
    assert storage["status"] == "OK"
    assert storage["free_bytes"] == free
    assert storage["usable_bytes_before_safety_reserve"] == 50 * 1024 ** 3
    assert storage["expected_raw_depth_bytes_per_second"] == raw_depth_rate
    assert storage["conservative_bytes_per_second"] == pytest.approx(expected_rate)
    assert storage["estimated_remaining_seconds"] == pytest.approx(
        50 * 1024 ** 3 / expected_rate,
    )


def test_real_time_tracking_default_expires_old_points_within_250ms():
    server = _load_collect_server()

    parser = server.build_parser()
    parsed = parser.parse_args([])

    assert parsed.max_tracking_age_ms == 250.0


def _closed_boundary_fixture():
    return {
        "camera_start_boundary_index": 9,
        "camera_end_boundary_proof": {
            "boundary_latched_atomically": True,
            "post_boundary_overwrite_prevented": True,
            "snapshot_frame_index": 12,
            "snapshot_age_s": 0.03,
            "max_snapshot_age_s": 0.1,
            "thread_alive_at_boundary": True,
            "thread_error_at_boundary": None,
        },
        "camera_delivery_fifo": {
            "status": "OK",
            "thread_alive": True,
            "frame_fresh": True,
            "interval_active": False,
            "accepting": False,
            "queue_depth": 0,
            "interval_overflow_frames": 0,
            "interval_enqueued_frames": 3,
            "interval_delivered_frames": 3,
            "first_enqueued_frame_index": 10,
            "last_enqueued_frame_index": 12,
            "last_delivered_frame_index": 12,
        },
        "capture_ingress_fifo": {
            "status": "OK",
            "thread_alive": True,
            "interval_active": False,
            "queue_depth": 0,
            "overflow_rejections": 0,
            "abandoned_after_error": 0,
            "accepted": 3,
            "processed": 3,
            "first_accepted_index": 10,
            "last_accepted_index": 12,
            "last_completed_index": 12,
        },
        "last_submitted_camera_index": 12,
        "last_completed_camera_index": 12,
    }


def test_end_boundary_closure_binds_snapshot_delivery_ingress_and_final_index():
    server = _load_collect_server()

    proof = server.validate_capture_boundary_closure(
        _closed_boundary_fixture(),
        n_rgb_frames=3,
        expected_fps=30.0,
        final_frame_range={"first_frame_index": 10, "last_frame_index": 12},
    )

    assert proof["closed"] is True
    assert proof["first_frame_index"] == 10
    assert proof["last_frame_index"] == 12


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("camera_end_boundary_proof", "thread_alive_at_boundary"), False, "thread"),
        (("camera_end_boundary_proof", "snapshot_age_s"), 0.11, "stale"),
        (("camera_delivery_fifo", "last_delivered_frame_index"), 11, "closure"),
        (("capture_ingress_fifo", "last_completed_index"), 11, "closure"),
        (("capture_ingress_fifo", "processed"), 2, "count"),
    ],
)
def test_end_boundary_closure_fails_closed_on_camera_freeze_or_index_gap(
    path, value, message,
):
    server = _load_collect_server()
    boundary = _closed_boundary_fixture()
    boundary[path[0]][path[1]] = value

    with pytest.raises(RuntimeError, match=message):
        server.validate_capture_boundary_closure(
            boundary,
            n_rgb_frames=3,
            expected_fps=30.0,
            final_frame_range={"first_frame_index": 10, "last_frame_index": 12},
        )


def test_complete_state_exposes_read_only_verified_video_evidence(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    session = tmp_path / "completed-session"
    session.mkdir()
    video_raw = b"video"
    (session / "rgb.mp4").write_bytes(video_raw)
    video_artifact = {
        "sha256": hashlib.sha256(video_raw).hexdigest(),
        "bytes": len(video_raw),
    }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": {"rgb.mp4": video_artifact},
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 321,
        "decoded_rgb_frames": 321,
        "diagnostic_finalization": {
            **seal,
        },
    }
    try:
        proof = collector.state_snapshot()["finalization_summary"]
        assert proof == {
            "complete": True,
            "rgb_mp4_path": str((session / "rgb.mp4").resolve()),
            "n_rgb_frames": 321,
            "decoded_rgb_frames": 321,
            "n_depth_frames": None,
            "rgb_mp4_sha256": video_artifact["sha256"],
            "diagnostic_seal_path": str(
                (session / "DIAGNOSTIC_FINALIZATION.json").resolve()
            ),
            "capture_boundary": None,
            "depth_integrity": None,
            "segment_validation": None,
            "timeline": None,
            "video_stitch": None,
                "video_timebase": None,
                "storage": None,
                "audio": None,
                "robot": None,
            }
    finally:
        collector.shutdown()


def test_complete_proof_accepts_missing_advisory_force_with_invalidity_flag(
    tmp_path,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    session = tmp_path / "completed-without-force"
    session.mkdir()
    video_raw = b"sealed video"
    (session / "rgb.mp4").write_bytes(video_raw)
    video_artifact = {
        "sha256": hashlib.sha256(video_raw).hexdigest(),
        "bytes": len(video_raw),
    }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": {"rgb.mp4": video_artifact},
        "force": None,
        "force_authenticity": None,
        "sealed_utc_ns": 1,
        "capture_boundary": {"force_advisory": [{"status": "DEGRADED"}]},
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n"
    )
    finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "force": None,
        "force_authenticity": None,
        "contact_force_valid": False,
        "capture_boundary": seal["capture_boundary"],
        "diagnostic_finalization": seal,
    }
    try:
        proof = collector._verified_finalization_summary(
            {"session_dir": str(session)}, finalization,
        )
        assert proof["complete"] is True
        assert proof["force"] is None
        assert proof["force_authenticity"] is None
        assert proof["contact_force_valid"] is False
    finally:
        collector.shutdown()


def test_completion_validator_accepts_missing_advisory_force(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    session = tmp_path / "validated-without-force"
    session.mkdir()
    depth_raw = b'{"frame_index":10}\n'
    (session / "depth_integrity.jsonl").write_bytes(depth_raw)
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"binary")
    collector.ffmpeg_path = str(ffmpeg)
    collector.ffmpeg_sha256 = hashlib.sha256(ffmpeg.read_bytes()).hexdigest()
    boundary = _closed_boundary_fixture()
    boundary["camera_delivery_start_proof"] = {
        "boundary_latched_atomically": True,
    }
    frame_range = {"first_frame_index": 10, "last_frame_index": 12}
    boundary["index_closure"] = server.validate_capture_boundary_closure(
        boundary,
        n_rgb_frames=3,
        expected_fps=collector._negotiated_camera_fps(),
        final_frame_range=frame_range,
    )
    result = {
        "session_dir": str(session),
        "n_rgb_frames": 3,
        "decoded_rgb_frames": 3,
        "n_depth_frames": 3,
        "video_timebase": {"verified": True},
        "depth_integrity": {
            "path": "depth_integrity.jsonl",
            "count": 3,
            "dtype": "uint16",
            "index_sha256": hashlib.sha256(depth_raw).hexdigest(),
        },
        "force": None,
        "force_authenticity": None,
        "contact_force_valid": False,
        "video_stitch": {
            "provenance": {
                "method": "ffmpeg_concat_demuxer_stream_copy",
                "sha256": collector.ffmpeg_sha256,
                "resolved_path": str(ffmpeg.resolve()),
                "shell": False,
            },
            "packet_validation": {
                "verified": True,
                "method": "ffmpeg_decode_null_muxer",
                "decoded_frame_count": 3,
                "expected_frame_count": 3,
            },
        },
        "capture_boundary": boundary,
        "frame_index_range": frame_range,
    }
    collector._verified_finalization_summary = (
        lambda _snapshot, _result: {"complete": True}
    )
    try:
        collector._validate_completion_result(result, session)
    finally:
        collector.shutdown()


def test_complete_video_proof_fails_closed_after_video_is_replaced(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    session = tmp_path / "completed-session"
    session.mkdir()
    original = b"sealed video"
    video = session / "rgb.mp4"
    video.write_bytes(original)
    artifact = {
        "sha256": hashlib.sha256(original).hexdigest(),
        "bytes": len(original),
    }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": {"rgb.mp4": artifact},
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "diagnostic_finalization": seal,
    }
    try:
        assert collector.state_snapshot()["finalization_summary"]["complete"] is True
        video.write_bytes(b"tampered")
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def test_complete_proof_fails_closed_after_sealed_audio_is_replaced(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        audio_device="Scarlett Solo USB: Audio",
        audio_sample_rate=48_000,
        audio_channels=2,
        audio_pick_channel=1,
        start_thread=False,
    )
    session = tmp_path / "completed-audio-session"
    session.mkdir()
    authenticity = {
        "schema_version": "wam_audio_signal_authenticity_v1",
        "policy_version": "audio_signal_degraded_real_v3",
        "status": "PASS",
    }
    files = {
        "rgb.mp4": b"sealed video",
        "audio.wav": b"sealed audio",
        "audio_capture.json": (
            json.dumps({"signal_authenticity": authenticity}).encode() + b"\n"
        ),
        "audio_input_times.jsonl": b"{}\n",
        "audio_times.jsonl": b"{}\n",
    }
    artifacts = {}
    for name, raw in files.items():
        (session / name).write_bytes(raw)
        artifacts[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    audio = {
        "path": "audio.wav",
        "status": "OK",
        "sample_rate": 16_000,
        "input_sample_rate": 48_000,
        "n_samples": 1_600,
        "sha256": artifacts["audio.wav"]["sha256"],
    }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": artifacts,
        "audio": audio,
        "audio_authenticity": authenticity,
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "audio": audio,
        "audio_authenticity": authenticity,
        "diagnostic_finalization": seal,
    }
    try:
        proof = collector.state_snapshot()["finalization_summary"]
        assert proof["complete"] is True
        assert proof["audio_authenticity"] == authenticity
        collector.platform._finalization["audio_authenticity"] = {
            **authenticity,
            "status": "FAIL",
        }
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        collector.platform._finalization["audio_authenticity"] = authenticity
        (session / "audio.wav").write_bytes(b"tampered")
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def test_complete_proof_binds_force_authenticity_and_rejects_receipt_change(
    tmp_path,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    session = tmp_path / "completed-force-session"
    session.mkdir()
    recorder = force_capture_module.ForceCaptureRecorder(
        session,
        t0=10.0,
        calibration_identity={
            "sha256": "c" * 64,
            "calibrated": True,
            "source_name": "force-test.json",
        },
        device_identity={
            "port": "/dev/serial/by-id/usb-test-hall",
            "baud": 115200,
            "serial_number": "TEST-HALL-01",
        },
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    )
    recorder.start()
    for index in range(501):
        raw = 31_000 + (index % 101)
        recorder.submit(force_sensor_module.ForceSample(
            t=10.0 + index * 0.01,
            raw=raw,
            hall_v=raw / 20_000.0,
            displacement_mm=1.25 + (index % 101) * 0.001,
            force_n=2.75 + (index % 101) * 0.001,
            status="WARN",
            device_t_ms=100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()
    produced = recorder.finish(t_end=15.0)
    force = {
        "path": "force.csv",
        "t_start": produced.t_start,
        "t_end": produced.t_end,
        "n_samples": produced.n_samples,
        "status": produced.status,
        "sha256": produced.sha256,
        "device_t_start_ms": produced.device_t_start_ms,
        "device_t_end_ms": produced.device_t_end_ms,
        "calibration_sha256": produced.calibration_sha256,
        "device_identity_sha256": produced.device_identity_sha256,
    }
    authenticity = produced.signal_authenticity
    (session / "rgb.mp4").write_bytes(b"sealed video")
    artifacts = {}
    for name in (
        "rgb.mp4",
        "force.csv",
        "force_capture.json",
        "FORCE_CAPTURE_COMPLETE.json",
    ):
        raw = (session / name).read_bytes()
        artifacts[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": artifacts,
        "force": force,
        "force_authenticity": authenticity,
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "force": force,
        "force_authenticity": authenticity,
        "diagnostic_finalization": seal,
    }
    try:
        proof = collector.state_snapshot()["finalization_summary"]
        assert proof["complete"] is True
        assert proof["force_authenticity"] == authenticity
        collector.platform._finalization["force_authenticity"] = {
            **authenticity,
            "status": "FAIL",
        }
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        collector.require_force = False
        collector.platform._finalization["force"] = "invalid"
        collector.platform._finalization["force_authenticity"] = "invalid"
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def test_complete_proof_rejects_uninventoried_pinned_force_calibration(
    tmp_path,
):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_force=True,
        start_thread=False,
    )
    session = tmp_path / "completed-calibrated-force-session"
    session.mkdir()
    calibration_payload = json.loads(
        (
            AUSCULTATION
            / "mechanical_sensor"
            / "force_feedback"
            / "force_feedback_calibration_veo6.json"
        ).read_text(encoding="utf-8")
    )
    calibration_payload["calibrated"] = True
    calibration_raw = (
        json.dumps(
            calibration_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    calibration_sha256 = hashlib.sha256(calibration_raw).hexdigest()
    calibration_identity = {
        "sha256": calibration_sha256,
        "calibrated": True,
        "source_name": "force-test-calibration.json",
        "artifact_schema_version": "wam_force_calibration_json_v1",
        "release_id": f"force-test-calibration-{calibration_sha256[:20]}",
    }
    calibration = force_sensor_module.ForceCalibration.from_bytes(calibration_raw)
    recorder = force_capture_module.ForceCaptureRecorder(
        session,
        t0=10.0,
        calibration_identity=calibration_identity,
        calibration_artifact=calibration_raw,
        device_identity={
            "port": "/dev/serial/by-id/usb-test-hall",
            "baud": 115200,
            "serial_number": "TEST-HALL-01",
        },
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    )
    recorder.start()
    for index in range(501):
        raw = 35_000 + (index % 201)
        recorder.submit(calibration.sample_from_raw(
            raw,
            10.0 + index * 0.01,
            device_t_ms=100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()
    produced = recorder.finish(t_end=15.0)
    force = {
        "path": "force.csv",
        "t_start": produced.t_start,
        "t_end": produced.t_end,
        "n_samples": produced.n_samples,
        "status": produced.status,
        "sha256": produced.sha256,
        "device_t_start_ms": produced.device_t_start_ms,
        "device_t_end_ms": produced.device_t_end_ms,
        "calibration_sha256": produced.calibration_sha256,
        "device_identity_sha256": produced.device_identity_sha256,
    }
    authenticity = produced.signal_authenticity
    assert authenticity["checks"]["calibration_recomputation"] == "PASS"
    (session / "rgb.mp4").write_bytes(b"sealed video")
    artifacts = {}
    # Deliberately omit force_calibration.json from the signed inventory.
    for name in (
        "rgb.mp4",
        "force.csv",
        "force_capture.json",
        "FORCE_CAPTURE_COMPLETE.json",
    ):
        raw = (session / name).read_bytes()
        artifacts[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": artifacts,
        "force": force,
        "force_authenticity": authenticity,
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "force": force,
        "force_authenticity": authenticity,
        "diagnostic_finalization": seal,
    }
    try:
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        calibration_bytes = (session / "force_calibration.json").read_bytes()
        artifacts["force_calibration.json"] = {
            "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "bytes": len(calibration_bytes),
        }
        (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
            json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
        )
        assert collector.state_snapshot()["finalization_summary"]["complete"] is True
        with (session / "force_calibration.json").open("ab") as handle:
            handle.write(b"tamper\n")
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def test_complete_proof_binds_full_quality_log_and_rejects_tamper(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    session = tmp_path / "completed-quality-session"
    session.mkdir()
    model_sha256 = "a" * 64
    calibration_sha256 = "c" * 64
    policy = {
        "schema_version": "wam_quality_policy_v1",
        "policy_id": "clinical-q-collector-test-v1",
        "force_band_n": [3.0, 5.0],
        "window_s": 0.25,
        "ema": 0.5,
        "max_audio_age_s": 0.2,
        "max_inference_age_s": 0.5,
        "max_force_age_s": 0.2,
        "min_packets_for_receipt": 2,
        "output_contract": {
            "model_role": "clinical_usability_q",
            "output_semantics": "class_probability",
            "positive_class": "usable",
            "calibration": {
                "method": "embedded_in_model",
                "release_id": (
                    f"clinical-q-calibration-test-{calibration_sha256[:20]}"
                ),
                "sha256": calibration_sha256,
            },
            "transform_to_p_usable": "identity",
            "output_range": [0.0, 1.0],
        },
    }
    force_recorder = force_capture_module.ForceCaptureRecorder(
        session,
        t0=7.2,
        calibration_identity={
            "sha256": "e" * 64,
            "calibrated": True,
            "source_name": "quality-force-test.json",
        },
        device_identity={
            "port": "/dev/serial/by-id/usb-quality-force-test",
            "baud": 115200,
            "serial_number": "QUALITY-FORCE-TEST",
        },
        max_device_gap_ms=400.0,
        max_host_gap_ms=400.0,
    )
    force_recorder.start()
    q_force_rows = {
        0: (31_000, 1.56, 1.20),
        30: (31_001, 1.561, 1.21),
        60: (31_002, 1.562, 1.22),
    }
    for index in range(501):
        raw, hall_v, displacement_mm = q_force_rows.get(
            index,
            (
                31_050 + (index % 101),
                1.60 + (index % 101) * 0.001,
                1.30 + (index % 101) * 0.001,
            ),
        )
        force_recorder.submit(force_sensor_module.ForceSample(
            t=7.2 + index * 0.01,
            raw=raw,
            hall_v=hall_v,
            displacement_mm=displacement_mm,
            force_n=(4.0 if index in q_force_rows else 4.0 + (index % 51) * 0.001),
            status="OK",
            device_t_ms=100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            force_recorder._queue.join()
    produced_force = force_recorder.finish(t_end=12.2)
    force = {
        "path": "force.csv",
        "t_start": produced_force.t_start,
        "t_end": produced_force.t_end,
        "n_samples": produced_force.n_samples,
        "status": produced_force.status,
        "sha256": produced_force.sha256,
        "device_t_start_ms": produced_force.device_t_start_ms,
        "device_t_end_ms": produced_force.device_t_end_ms,
        "calibration_sha256": produced_force.calibration_sha256,
        "device_identity_sha256": produced_force.device_identity_sha256,
    }
    force_authenticity = produced_force.signal_authenticity
    writer = realtime_q.QualityPacketLogWriter(session)
    runtime = realtime_q.StreamingQuality(
        scorer=lambda samples, _sr: float(np.mean(samples)),
        mode="production",
        model_release_id=f"clinical-q-collector-test-{model_sha256[:20]}",
        model_sha256=model_sha256,
        policy=policy,
        packet_log_writer=writer,
    )
    for sequence, value, inference_t in (
        (30, 0.1, 7.25),
        (31, 0.2, 7.55),
        (32, 0.3, 7.85),
    ):
        runtime.update(
            audio_window=server.SimpleNamespace(
                samples=np.linspace(
                    value, value + 0.01, 4_000, dtype=np.float32
                ),
                sr=16_000,
                t_start=inference_t - 0.25,
                t_end=inference_t,
                status="OK",
                raw_source={
                    "source_channel": 1,
                    "source_sample_start": int(round((inference_t - 0.25) * 48_000)),
                    "source_sample_end": int(round((inference_t - 0.25) * 48_000)) + 12_000,
                    "source_sample_rate": 48_000,
                    "resampler_identity": {
                        "algorithm": "polyphase_fir_v1",
                        "input_sample_rate": 48_000,
                        "output_sample_rate": 16_000,
                    },
                },
            ),
            force_sample=server.SimpleNamespace(
                t=inference_t - 0.05,
                raw=31_000 + (sequence - 30),
                hall_v=1.56 + (sequence - 30) * 0.001,
                displacement_mm=1.20 + (sequence - 30) * 0.01,
                force_n=4.0,
                status="OK",
                device_t_ms=100.0 + (sequence - 30) * 300.0,
            ),
            inference_sequence=sequence,
            inference_monotonic_s=inference_t,
            observed_monotonic_s=inference_t + 0.01,
        )
    q_authenticity = runtime.quality_authenticity(now_monotonic_s=7.9)
    log_receipt = writer.finalize(q_authenticity, now_monotonic_s=7.9)
    quality = realtime_q.quality_packet_log_summary(log_receipt)
    (session / "rgb.mp4").write_bytes(b"sealed video")
    artifacts = {}
    artifact_names = [
        "rgb.mp4",
        "force.csv",
        "force_capture.json",
        "FORCE_CAPTURE_COMPLETE.json",
        "quality_inference_packets.jsonl",
        "quality_inference_authenticity.json",
    ]
    artifact_names.extend(
        claim["path"]
        for claim in log_receipt["model_input_inventory"]["files"]
    )
    for name in artifact_names:
        raw = (session / name).read_bytes()
        artifacts[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": artifacts,
        "force": force,
        "force_authenticity": force_authenticity,
        "quality": quality,
        "quality_authenticity": log_receipt,
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "force": force,
        "force_authenticity": force_authenticity,
        "quality": quality,
        "quality_authenticity": log_receipt,
        "diagnostic_finalization": seal,
    }
    try:
        proof = collector.state_snapshot()["finalization_summary"]
        assert proof["complete"] is True
        assert proof["quality"] == quality
        collector.platform._finalization["quality_authenticity"] = {
            **log_receipt,
            "status": "FAIL",
        }
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        collector.platform._finalization["quality"] = "invalid"
        collector.platform._finalization["quality_authenticity"] = "invalid"
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        collector.platform._finalization["quality"] = quality
        collector.platform._finalization["quality_authenticity"] = log_receipt
        quality_input = (
            session
            / "quality_model_inputs"
            / "input_00000000000000000031.f32le"
        )
        original_quality_input = quality_input.read_bytes()
        quality_input.write_bytes(b"tampered-model-input")
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
        quality_input.write_bytes(original_quality_input)
        assert collector.state_snapshot()["finalization_summary"]["complete"] is True
        with (session / "quality_inference_packets.jsonl").open("ab") as handle:
            handle.write(b'{"tampered":true}\n')
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def _v3_robot_summary(*, frame_count=1, control_tick_count=1, **changes):
    summary = {
        "schema": "wam_openarm_exact_control_capture_summary_v3",
        "status": "exact_control_capture_validated_socketcan_tx_enqueue",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "frame_count": frame_count,
        "control_tick_count": control_tick_count,
        "send_status_observable": True,
        "send_status_reason": "openarm_can_1_2_2_full_frame_write_results",
        "send_status_semantics": (
            "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
        ),
        "motor_ack_observable": False,
        "motor_ack_reason": (
            "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
        ),
        "feedback_liveness_observable": False,
        "feedback_liveness_reason": (
            "v3_has_no_per_motor_feedback_rx_counters_or_timestamps"
        ),
        "training_eligible": False,
        "eligibility_blockers": ["robot_feedback_liveness_unobservable"],
        "arm_tx_expected_frame_count": control_tick_count * 7,
        "arm_tx_full_frame_write_count": control_tick_count * 7,
        "arm_tx_failed_frame_write_count": 0,
        "gripper_tx_expected_frame_count": control_tick_count,
        "gripper_tx_full_frame_write_count": control_tick_count,
        "gripper_tx_failed_frame_write_count": 0,
        "socketcan_tx_enqueue_failure_tick_count": 0,
    }
    summary.update(changes)
    return summary


def _robot_contract():
    from ausc.perception.robot.robot_state_stream import RobotStateContract

    return RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=tuple(
            f"openarm_left_joint{index}" for index in range(1, 8)
        ),
        joint_position_limits_rad=tuple((-4.0, 4.0) for _ in range(7)),
        leader_can_interface="can1",
        follower_can_interface="can3",
        urdf_sha256="a" * 64,
        joint_mapper_sha256="b" * 64,
        camera_profile_sha256="c" * 64,
        timing_artifact_sha256="d" * 64,
        pair_contract_sha256="e" * 64,
        controller_binary_sha256="f" * 64,
        controller_source_manifest_sha256="1" * 64,
        follower_config_sha256="2" * 64,
        openarm_sdk_sha256="3" * 64,
        max_abs_joint_velocity_rad_s=8.0,
        gripper_motor_position_limits_rad=(-0.2, 0.2),
    )


def _v4_robot_summary(*, frame_count=1, control_tick_count=1, **changes):
    activated_contract = _robot_contract()
    summary = _v3_robot_summary(
        frame_count=frame_count,
        control_tick_count=control_tick_count,
    )
    summary.update({
        "schema": "wam_openarm_exact_control_capture_summary_v4",
        "status": "exact_control_capture_validated_feedback_liveness",
        "feedback_liveness_observable": True,
        "feedback_liveness_reason": (
            "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
        ),
        "feedback_liveness_semantics": (
            "counter_generation_increment_only_after_matching_motor_id_valid_state_decode"
        ),
        "feedback_max_age_ns": 20_000_000,
        "feedback_max_gap_ns": 20_000_000,
        "feedback_counter_progress_all_motors": True,
        "feedback_all_ticks_fresh": True,
        "feedback_gap_clean": True,
        "feedback_clean": True,
        "feedback_gap_violation_count_delta": [0] * 8,
        "trajectory_validation_schema": "wam_openarm_trajectory_validation_v1",
        "trajectory_nondegenerate": True,
        "command_response_consistent": True,
        "command_response_violation_joint_names": [],
        "training_eligible": True,
        "eligibility_blockers": [],
        "buffer_duration_ns": activated_contract.buffer_duration_ns,
        "max_packets": activated_contract.max_packets,
        "max_gap_ns": activated_contract.max_gap_ns,
        "max_staleness_ns": activated_contract.max_staleness_ns,
        "max_exposure_interval_ns": activated_contract.max_exposure_interval_ns,
    })
    summary.update(changes)
    return summary


def test_historical_v3_robot_receipt_remains_diagnostic_only():
    server = _load_collect_server()
    summary = _v3_robot_summary(frame_count=1, control_tick_count=1)

    assert server._robot_v3_tx_summary_valid(summary) is True
    assert server._robot_v4_summary_valid(summary, contract=_robot_contract()) is False


def test_complete_proof_binds_all_left_robot_artifacts_and_detects_replacement(
    tmp_path,
):
    server = _load_collect_server()
    coordinator = server.SimpleNamespace(
        faulted=False,
        fault_reason="",
        contract=_robot_contract(),
        telemetry_schema="wam_openarm_control_envelope_v4",
        telemetry_schemas=frozenset({"wam_openarm_control_envelope_v4"}),
    )
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=coordinator,
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    session = tmp_path / "completed-robot-session"
    session.mkdir()
    files = {
        "rgb.mp4": b"sealed video",
        "robot_control_envelopes.jsonl": b'{"schema":"wam_openarm_control_envelope_v4"}\n',
        "robot_exact_control.csv": b"schema,control_tick_seq\nwam_openarm_exact_control_record_v4,1\n",
        "robot_frame_index.jsonl": b'{"schema":"wam_openarm_exact_control_frame_evidence_v4","frame_index":1}\n',
    }
    artifacts = {}
    for name, raw in files.items():
        (session / name).write_bytes(raw)
        artifacts[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    robot = _v4_robot_summary(
        hash_inventory={
            name: {
                "sha256": artifacts[name]["sha256"],
                "size_bytes": artifacts[name]["bytes"],
            }
            for name in (
                "robot_control_envelopes.jsonl",
                "robot_exact_control.csv",
                "robot_frame_index.jsonl",
            )
        },
    )
    robot_summary_raw = (
        json.dumps(robot, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (session / "robot_capture_summary.json").write_bytes(robot_summary_raw)
    artifacts["robot_capture_summary.json"] = {
        "sha256": hashlib.sha256(robot_summary_raw).hexdigest(),
        "bytes": len(robot_summary_raw),
    }
    manifest_raw = b'{"status":"diagnostic_complete"}\n'
    (session / "manifest.json").write_bytes(manifest_raw)
    seal = {
        "schema_version": "wam_diagnostic_finalization_v1",
        "capture_mode": "diagnostic_perception_only",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "artifacts": artifacts,
        "robot": robot,
        "sealed_utc_ns": 1,
        "capture_boundary": None,
    }
    (session / "DIAGNOSTIC_FINALIZATION.json").write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
    )
    collector.platform.session_dir = session
    collector.platform.session_id = session.name
    collector.platform.state = server.CollectionState.COMPLETE
    collector.platform._finalization = {
        "n_rgb_frames": 1,
        "decoded_rgb_frames": 1,
        "robot": robot,
        "diagnostic_finalization": seal,
    }
    try:
        proof = collector.state_snapshot()["finalization_summary"]
        assert proof["complete"] is True
        assert proof["robot"]["frame_count"] == 1
        (session / "robot_exact_control.csv").write_bytes(b"tampered")
        assert collector.state_snapshot()["finalization_summary"]["complete"] is False
    finally:
        collector.shutdown()


def test_page_allows_a_fresh_session_after_complete_or_abort():
    server = _load_collect_server()
    page = server.PAGE.decode("utf-8")

    # The Start button defers to the single platform authority (start_authorized)
    # rather than recomputing the startable-state set; COMPLETE/ABORTED/ERROR
    # (which are startable) surface a fresh-capture affordance.
    assert "s.start_authorized!==true" in page
    assert "New capture" in page
    assert "latest?.state==='RECORDING'" in page


def test_collector_shutdown_aborts_an_open_diagnostic_session(tmp_path):
    server = _load_collect_server()
    targets = [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))]
    collector = server.Collector(
        targets, fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    assert collector.control("start")["ok"] is True
    session_dir = collector.platform.session_dir

    collector.shutdown()

    assert collector.platform.state.value == "ABORTED"
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert "service shutdown" in manifest["abort_reason"]
    assert not (session_dir / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_automatic_abort_freezes_elapsed_time(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._record_started = time.monotonic() - 1.0
        collector._abort_recording("camera stale")
        first = collector.state_snapshot()["elapsed"]
        time.sleep(0.05)
        second = collector.state_snapshot()["elapsed"]
        assert collector.platform.state.value == "ABORTED"
        assert first == second
    finally:
        collector.shutdown()


def test_idle_runtime_fault_clears_back_confirmation(tmp_path):
    server = _load_collect_server()

    class BackGate:
        cleared = []

        @classmethod
        def clear_confirmation(cls, reason):
            cls.cleared.append(reason)
            return {"active": False, "confirmed": False}

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._back_gate = BackGate()
    try:
        assert collector.platform.state.value == "READY"
        assert collector._abort_recording("idle model failed") is False
        assert BackGate.cleared == ["RUNTIME_FAULT"]
    finally:
        collector.shutdown()


class _IngressStub:
    def __init__(self):
        self.submitted = []
        self.bundles = []
        self.drain_calls = 0
        self.dropped = 0
        self.overflow = 0
        self.begin_calls = 0
        self.close_calls = 0
        self.drain_targets = []
        self.last_completed_index = -1

    def begin_interval(self):
        self.begin_calls += 1
        return True

    def submit(self, bundle):
        self.submitted.append(bundle.index)
        self.bundles.append(bundle)
        return True

    def close_interval(self):
        self.close_calls += 1
        return self.submitted[-1] if self.submitted else None

    def drain(self, timeout=0, through_index=None):
        self.drain_calls += 1
        self.drain_targets.append(through_index)
        if through_index is not None:
            self.last_completed_index = through_index
        return True

    def health(self):
        return {
            "status": "OK",
            "error": None,
            "dropped_before_inference": self.dropped,
            "overflow_rejections": self.overflow,
            "outside_interval_rejections": 0,
            "last_completed_index": self.last_completed_index,
            "queue_depth": 0,
        }

    def stop(self):
        pass


def test_robot_upper_bracket_backpressures_the_same_camera_frame_without_reordering(
    tmp_path,
):
    server = _load_collect_server()
    now = time.monotonic()
    calls = []

    class Robot:
        faulted = False
        fault_reason = ""

        def build_frame_evidence_inputs(
            self, *, frame_index, camera_host_t_s, camera_device_t_ms,
        ):
            calls.append(frame_index)
            if calls == [1]:
                raise server.RobotCaptureNotReady("upper bracket pending")
            exposure = server.SimpleNamespace(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                camera_device_t_ms=camera_device_t_ms,
                camera_id="CL8326300A0",
                camera_profile_sha256="a" * 64,
                alignment_sha256="b" * 64,
                camera_latency_sha256="c" * 64,
            )
            return server.SimpleNamespace(
                camera_exposure=exposure,
                robot_trajectory=object(),
                persisted_index={
                    "exact_action_tick_first": frame_index * 10,
                    "exact_action_tick_last": frame_index * 10 + 2,
                },
            )

    mask_calls = []

    class MaskProvider:
        @staticmethod
        def build(*, exposure, trajectory, aligned_depth=None,
                  depth_scale=1.0):
            del trajectory
            mask_calls.append((aligned_depth, depth_scale))
            return server.SimpleNamespace(
                frame_index=exposure.frame_index,
                mask_sha256="d" * 64,
                geometry_release_id="openarm_geometry-" + "1" * 20,
                tool_release_id="openarm_tool-" + "2" * 20,
                handeye_release_id="handeye_activation-" + "9" * 20,
                handeye_sha256="9" * 64,
                handeye_capability_schema=(
                    "wam_provisional_robot_mask_handeye_capability_v1"
                ),
                handeye_calibration_tier="USER_ACCEPTED_BELOW_GATE_5MM",
                handeye_policy_sha256="8" * 64,
                handeye_calibration_status="USER_ACCEPTED_BELOW_GATE",
                handeye_provisional=True,
                handeye_quality_gate_passed=False,
                handeye_workspace_1sigma_max_m=0.00605,
                handeye_covariance_bound_transform=(
                    "transform_follower_base_from_camera"
                ),
                handeye_covariance_scope="random_only",
                handeye_systematic_bias_status="NOT_CHARACTERIZED",
                handeye_systematic_bias_bound_m=None,
                handeye_autonomous_targeting_authorized=False,
                handeye_provisional_policy_allowance_m=0.0065,
            )

    def build_bundle(**kwargs):
        frame = kwargs["camera_frame"]
        canonical = json.dumps(
            {
                "frame_index": frame.index,
                "schema": "wam_frame_evidence_bundle_v3",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return server.SimpleNamespace(
            index=frame.index,
            t=frame.t,
            camera_frame=frame,
            bundle_id=(
                "frame_evidence-" + hashlib.sha256(canonical).hexdigest()
            ),
            canonical_metadata_bytes=lambda canonical=canonical: canonical,
            robot_trajectory_sha256="e" * 64,
            robot_exact_action_sha256="f" * 64,
            robot_exact_action_tick_first=frame.index * 10,
            robot_exact_action_tick_last=frame.index * 10 + 2,
            robot_exact_action_count=3,
            robot_mask_packet=kwargs["robot_mask_packet"],
        )

    class Camera:
        def __init__(self):
            self.queue = [
                server.SimpleNamespace(
                    index=index,
                    t=now + index * 0.01,
                    status="OK",
                    frame=np.zeros((2, 2, 3), dtype=np.uint8),
                    depth=np.ones((2, 2), dtype=np.uint16),
                    depth_scale=1.0,
                    device_t_ms=float(index),
                    time_source="test",
                )
                for index in (1, 2)
            ]

        def drain_delivery(self, max_frames=1):
            batch, self.queue = self.queue[:max_frames], self.queue[max_frames:]
            return tuple(batch)

        def delivery_health(self):
            return {
                "status": "OK",
                "interval_active": True,
                "accepting": True,
                "queue_depth": len(self.queue),
                "interval_overflow_frames": 0,
                "overflow_frames": 0,
            }

        def abort_delivery_interval(self):
            self.queue.clear()

        def stop(self):
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=MaskProvider(),
        frame_evidence_bundle_factory=build_bundle,
        start_thread=False,
    )
    collector._ingress_worker.stop()
    ingress = _IngressStub()
    collector._ingress_worker = ingress
    collector._cam = Camera()
    collector.platform.state = server.CollectionState.RECORDING
    collector._capture_ingress_paused = False
    collector._camera_delivery_enabled = True
    collector._recording_boundary = {
        "start_host_t_s": now,
        "camera_start_boundary_index": 0,
        "model_start_boundary_index": -1,
    }
    try:
        assert collector._drain_camera_delivery() == 0
        assert collector._pending_delivery_frame.index == 1
        assert [frame.index for frame in collector._cam.queue] == [2]
        assert ingress.submitted == []

        assert collector._drain_camera_delivery() == 2
        assert collector._pending_delivery_frame is None
        assert collector._cam.queue == []
        assert ingress.submitted == [1, 2]
        assert calls == [1, 1, 2]
        # Depth-veto wiring: the SAME frame's aligned depth reaches the
        # mask provider with its scale, enabling the physically-proven
        # arm-free pixel veto on the recorded-evidence path.
        assert len(mask_calls) == 2
        for aligned_depth, depth_scale in mask_calls:
            assert isinstance(aligned_depth, np.ndarray)
            assert aligned_depth.dtype == np.uint16
            assert aligned_depth.shape == (2, 2)
            assert depth_scale == 1.0
        ledger = ingress.bundles[0].tracking["frame_evidence"]
        assert ingress.bundles[0].tracking["frame_evidence_bundle_id"] == (
            ledger["bundle_id"]
        )
        assert ledger["schema"] == "wam_frame_evidence_ledger_record_v1"
        canonical = ledger["canonical_json"].encode("ascii")
        assert hashlib.sha256(canonical).hexdigest() == (
            ledger["canonical_sha256"]
        )
        assert ledger["bundle_id"] == (
            "frame_evidence-" + ledger["canonical_sha256"]
        )
        assert json.loads(ledger["canonical_json"])["schema"] == (
            "wam_frame_evidence_bundle_v3"
        )
        assert ingress.bundles[0].tracking["robot_pair"] == "left:can1->can3"
        handeye = ingress.bundles[0].tracking["robot_mask"]
        assert handeye["handeye_calibration_status"] == (
            "USER_ACCEPTED_BELOW_GATE"
        )
        assert handeye["handeye_provisional"] is True
        assert handeye["handeye_capability_schema"] == (
            "wam_provisional_robot_mask_handeye_capability_v1"
        )
        assert handeye["handeye_calibration_tier"] == (
            "USER_ACCEPTED_BELOW_GATE_5MM"
        )
        assert handeye["handeye_policy_sha256"] == "8" * 64
        assert handeye["handeye_autonomous_targeting_authorized"] is False
    finally:
        collector.platform.state = server.CollectionState.ABORTED
        collector.shutdown()


def test_robot_sidecar_is_finalized_and_bound_before_platform_finalize(tmp_path):
    server = _load_collect_server()
    calls = []
    summary = _v4_robot_summary(frame_count=3, control_tick_count=3)
    activated_contract = _robot_contract()

    class Robot:
        faulted = False
        fault_reason = ""
        telemetry_schema = "wam_openarm_control_envelope_v4"
        telemetry_schemas = frozenset({telemetry_schema})
        contract = activated_contract

        @staticmethod
        def finish_session():
            calls.append("robot_finish")
            return summary

    class Capture:
        def request_cancel(self):
            pass

        @staticmethod
        def set_robot_summary(value, *, contract):
            calls.append("capture_bind_robot")
            assert value == summary
            assert contract is Robot.contract

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.capture = Capture()
    try:
        result = collector._finish_robot_session(expected_frame_count=3)
        assert result == summary
        assert calls == ["robot_finish", "capture_bind_robot"]
        assert collector._robot_status == "READY"
        assert collector._robot_last_summary == summary
    finally:
        collector.platform.capture = None
        collector.shutdown()


def test_production_robot_finish_rejects_historical_v3_summary(tmp_path):
    server = _load_collect_server()

    class Robot:
        faulted = False
        fault_reason = ""
        telemetry_schema = "wam_openarm_control_envelope_v4"
        telemetry_schemas = frozenset({telemetry_schema})
        contract = _robot_contract()

        @staticmethod
        def finish_session():
            return _v3_robot_summary(frame_count=3, control_tick_count=3)

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.capture = server.SimpleNamespace(
        set_robot_summary=lambda _summary, *, contract: None,
    )
    try:
        with pytest.raises(RuntimeError, match="v4.*feedback liveness"):
            collector._finish_robot_session(expected_frame_count=3)
    finally:
        collector.platform.capture = None
        collector.shutdown()


@pytest.mark.parametrize("contract", [None, SimpleNamespace(side="left")])
def test_production_robot_finish_requires_actual_frozen_contract(
    tmp_path, contract,
):
    server = _load_collect_server()

    class Robot:
        faulted = False
        fault_reason = ""

        @staticmethod
        def finish_session():
            pytest.fail("invalid activation contract must fail before End")

    robot = Robot()
    robot.contract = contract
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=robot,
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    try:
        with pytest.raises(RuntimeError, match="actual frozen RobotStateContract"):
            collector._finish_robot_session(expected_frame_count=3)
    finally:
        collector.shutdown()


def test_realtime_robot_ready_gate_rejects_non_v4_coordinator(tmp_path):
    server = _load_collect_server()

    class Robot:
        faulted = False
        fault_reason = ""
        recording = False
        camera_arm_authorized = False
        contract = _robot_contract()
        telemetry_schema = "wam_openarm_control_envelope_v3"
        telemetry_schemas = frozenset({telemetry_schema})

        @staticmethod
        def receiver_health():
            return {
                "status": "READY",
                "preflight_ready": True,
                "pair": "left:can1->can3",
            }

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    try:
        snapshot = collector._robot_snapshot()
        assert snapshot["ready"] is False
        assert "v4" in snapshot["error"]
    finally:
        collector.shutdown()


def test_robot_sidecar_frame_count_must_equal_rgbd_ingress_count(tmp_path):
    server = _load_collect_server()

    class Robot:
        faulted = False
        fault_reason = ""
        contract = _robot_contract()

        @staticmethod
        def finish_session():
            return _v4_robot_summary(frame_count=2, control_tick_count=2)

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.capture = server.SimpleNamespace(
        set_robot_summary=lambda _summary, *, contract: None,
    )
    try:
        with pytest.raises(RuntimeError, match="robot/RGB-D frame count"):
            collector._finish_robot_session(expected_frame_count=3)
    finally:
        collector.platform.capture = None
        collector.shutdown()


def test_robot_sidecar_rejects_any_socketcan_tx_enqueue_failure(tmp_path):
    server = _load_collect_server()

    class Robot:
        faulted = False
        fault_reason = ""
        contract = _robot_contract()

        @staticmethod
        def finish_session():
            return _v4_robot_summary(
                frame_count=3,
                control_tick_count=3,
                arm_tx_full_frame_write_count=20,
                arm_tx_failed_frame_write_count=1,
                socketcan_tx_enqueue_failure_tick_count=1,
                training_eligible=False,
                eligibility_blockers=[
                    "robot_socketcan_tx_enqueue_failure",
                ],
            )

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.capture = server.SimpleNamespace(
        set_robot_summary=lambda _summary, *, contract: None,
    )
    try:
        with pytest.raises(RuntimeError, match="SocketCAN TX"):
            collector._finish_robot_session(expected_frame_count=3)
    finally:
        collector.platform.capture = None
        collector.shutdown()


def test_live_model_uses_same_frame_robot_mask_bundle_before_recording(tmp_path):
    server = _load_collect_server()
    frame = server.SimpleNamespace(
        index=7,
        t=time.monotonic(),
        status="OK",
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.uint16),
        depth_scale=1.0,
        device_t_ms=7.0,
        time_source="test",
    )
    calls = []

    class Robot:
        faulted = False
        fault_reason = ""

        @staticmethod
        def build_live_frame_evidence_inputs(
            *, frame_index, camera_host_t_s, camera_device_t_ms,
        ):
            calls.append(("live", frame_index))
            return server.SimpleNamespace(
                camera_exposure=server.SimpleNamespace(
                    frame_index=frame_index,
                    camera_host_t_s=camera_host_t_s,
                    camera_device_t_ms=camera_device_t_ms,
                    camera_id="CL8326300A0",
                    camera_profile_sha256="a" * 64,
                    alignment_sha256="b" * 64,
                    camera_latency_sha256="c" * 64,
                ),
                robot_trajectory=object(),
            )

    class MaskProvider:
        @staticmethod
        def build(*, exposure, trajectory, aligned_depth=None,
                  depth_scale=1.0):
            del trajectory
            assert isinstance(aligned_depth, np.ndarray)
            assert aligned_depth.dtype == np.uint16
            assert depth_scale == 1.0
            calls.append(("mask", exposure.frame_index))
            return server.SimpleNamespace(frame_index=exposure.frame_index)

    expected = server.SimpleNamespace(
        index=7,
        camera_frame=frame,
        robot_mask_packet=server.SimpleNamespace(frame_index=7),
    )

    def build_bundle(**kwargs):
        calls.append(("bundle", kwargs["camera_frame"].index))
        expected.robot_mask_packet = kwargs["robot_mask_packet"]
        return expected

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=Robot(),
        robot_mask_provider=MaskProvider(),
        frame_evidence_bundle_factory=build_bundle,
        start_thread=False,
    )
    try:
        assert collector._model_input_for_frame(frame) is expected
        assert calls == [("live", 7), ("mask", 7), ("bundle", 7)]
    finally:
        collector.shutdown()


def test_recording_model_waits_for_the_persisted_same_frame_robot_bundle(tmp_path):
    server = _load_collect_server()
    frame = server.SimpleNamespace(
        index=8, t=time.monotonic(), device_t_ms=8.0,
    )
    coordinator = server.SimpleNamespace(
        faulted=False,
        fault_reason="",
        build_live_frame_evidence_inputs=lambda **_kwargs: pytest.fail(
            "recording must not query non-persisted preview evidence"
        ),
    )
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=coordinator,
        robot_mask_provider=object(),
        frame_evidence_bundle_factory=lambda **_kwargs: object(),
        start_thread=False,
    )
    collector.platform.state = server.CollectionState.RECORDING
    try:
        with pytest.raises(server.RobotCaptureNotReady, match="persisted"):
            collector._model_input_for_frame(frame)
        evidence = server.SimpleNamespace(
            index=8, camera_frame=frame, robot_mask_packet=object(),
        )
        collector._latest_recorded_robot_evidence = evidence
        assert collector._model_input_for_frame(frame) is evidence

        # 2026-08-14: the model consumes the NEWEST evidence regardless of the
        # pending camera frame's index — the exact-index rendezvous starved
        # inference to ~4 submissions per 3-minute recording.
        newer_evidence = server.SimpleNamespace(
            index=11, camera_frame=frame, robot_mask_packet=object(),
        )
        collector._latest_recorded_robot_evidence = newer_evidence
        # Inference throttle (2026-08-14): immediate re-offers inside the
        # minimum interval are refused so back-to-back ~750 ms inferences
        # cannot starve the recorded-evidence thread's GIL share...
        with pytest.raises(server.RobotCaptureNotReady, match="throttle"):
            collector._model_input_for_frame(frame)
        # ...and once the interval elapses the NEWEST evidence flows.
        collector._recording_model_submit_monotonic -= (
            server._RECORDING_MODEL_MIN_SUBMIT_INTERVAL_S + 0.01
        )
        assert collector._model_input_for_frame(frame) is newer_evidence
    finally:
        collector.platform.state = server.CollectionState.ABORTED
        collector.shutdown()


def test_capture_ingress_is_armed_only_inside_explicit_recording_interval(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    frame = server.SimpleNamespace(index=8, t=1.0)
    try:
        assert collector._submit_ingress(frame, {}) is True
        assert stub.submitted == []
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        frame.t = collector._recording_boundary["start_host_t_s"] + 0.01
        assert collector._submit_ingress(frame, {}) is True
        assert stub.submitted == [8]
        result = collector.control("end")
        assert result["ok"] is True
        assert stub.begin_calls == 1
        assert stub.close_calls == 1
        assert stub.drain_targets == [8]
        assert collector._submit_ingress(server.SimpleNamespace(
            index=9, t=collector._recording_boundary["start_host_t_s"] + 0.02,
        ), {}) is True
        assert stub.submitted == [8]
    finally:
        if collector.platform.state.value == "RECORDING":
            collector.platform.abort("test cleanup")
        collector.shutdown()


def test_direct_camera_path_cannot_submit_after_lossless_delivery_is_armed(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        collector._camera_delivery_enabled = True
        boundary_t = collector._recording_boundary["start_host_t_s"]
        frame = server.SimpleNamespace(
            index=1, t=boundary_t + 0.01, device_t_ms=10.0,
        )

        assert collector._submit_camera_capture(frame, source="direct") is True
        assert stub.submitted == []

        assert collector._submit_camera_capture(frame, source="delivery") is True
        assert stub.submitted == [1]
    finally:
        collector._abort_recording("test cleanup")
        collector.shutdown()


def test_model_staleness_records_unavailable_association_without_aborting_rgbd(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    collector._model_worker = SimpleNamespace(health=lambda: {
        "status": "OK", "thread_alive": True, "last_result_age_s": 5.0,
    })
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        boundary_t = collector._recording_boundary["start_host_t_s"]
        collector._latest_tracking = {
            "frame_index": 1, "camera_host_t_s": boundary_t + 0.001,
            "back_present": True, "initialized": True, "reason": "TRACKING",
            "points": [],
        }

        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=1, t=boundary_t + 0.01, device_t_ms=10.0,
        )) is True

        assert collector.platform.state.value == "RECORDING"
        assert stub.bundles[-1].tracking["tracking_association"] == "unavailable"
        assert stub.bundles[-1].tracking["reason"] == "MODEL_UNAVAILABLE_FOR_CAMERA_FRAME"
    finally:
        collector._model_worker = None
        collector._abort_recording("test cleanup")
        collector.shutdown()


def test_end_aborts_instead_of_sealing_when_capture_ingress_dropped_a_frame(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        stub.dropped = 1

        result = collector.control("end")

        assert result["ok"] is False
        assert result["code"] == "capture_ingress_not_drained"
        assert result["state"] == "ABORTED"
        assert "dropped 1 frame" in result["message"]
        assert stub.drain_calls == 1
    finally:
        collector.shutdown()


def test_end_boundary_blocks_concurrent_operator_controls_until_finalizing(tmp_path):
    server = _load_collect_server()
    entered = threading.Event()
    release = threading.Event()

    class BlockingIngress(_IngressStub):
        def drain(self, timeout=0, through_index=None):
            self.drain_calls += 1
            self.drain_targets.append(through_index)
            entered.set()
            assert release.wait(1.0)
            return True

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = BlockingIngress()
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    assert collector.control("start")["ok"] is True
    collector._current_tracking = {
        "constellation_valid": True, "reason": "TRACKING",
    }
    collector.platform.back_present = True
    collector.platform.initialized = True
    collector.platform.target_valid = True
    collector.platform.tracking_points = [{"name": "R1", "status": "TRACKING"}]
    results = {}

    end_thread = threading.Thread(
        target=lambda: results.setdefault("end", collector.control("end")), daemon=True,
    )
    confirm_thread = threading.Thread(
        target=lambda: results.setdefault("confirm", collector.control("confirm")), daemon=True,
    )
    try:
        end_thread.start()
        assert entered.wait(1.0)
        confirm_thread.start()
        time.sleep(0.05)
        assert "confirm" not in results
        release.set()
        end_thread.join(1.0)
        confirm_thread.join(1.0)
        assert results["end"]["ok"] is True
        assert results["confirm"]["ok"] is False
        assert results["confirm"]["code"] == "not_recording"
    finally:
        release.set()
        collector.shutdown()


def test_live_fault_during_end_drain_cannot_abort_closed_capture_boundary(tmp_path):
    server = _load_collect_server()
    entered = threading.Event()
    release = threading.Event()

    class BlockingIngress(_IngressStub):
        def drain(self, timeout=0, through_index=None):
            self.drain_calls += 1
            entered.set()
            assert release.wait(1.0)
            return True

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = BlockingIngress()
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    assert collector.control("start")["ok"] is True
    result = {}
    end_thread = threading.Thread(
        target=lambda: result.setdefault("end", collector.control("end")), daemon=True,
    )
    try:
        end_thread.start()
        assert entered.wait(1.0)
        collector._abort_recording("model became stale while End was draining")
        assert collector.platform.state.value == "RECORDING"
        release.set()
        end_thread.join(1.0)
        assert result["end"]["ok"] is True
        assert collector.platform.state.value in {"FINALIZING", "COMPLETE"}
    finally:
        release.set()
        collector.shutdown()


def test_collector_submits_every_camera_frame_with_explicit_tracking_source(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    collector._latest_tracking = {
        "frame_index": 7,
        "camera_host_t_s": 100.0,
        "camera_device_t_ms": 2000.0,
        "back_present": True,
        "initialized": True,
        "reason": "TRACKING",
        "points": [],
    }
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        start_t = collector._recording_boundary["start_host_t_s"]
        collector._latest_tracking["camera_host_t_s"] = start_t + 0.001
        for index in (10, 11, 12):
            frame = server.SimpleNamespace(
                index=index, t=start_t + (index - 9) / 30.0,
                device_t_ms=2000.0 + index * 33.0,
            )
            assert collector._submit_camera_capture(frame) is True

        assert stub.submitted == [10, 11, 12]
        assert [bundle.tracking["source_tracking_frame_index"] for bundle in stub.bundles] == [
            7, 7, 7,
        ]
    finally:
        collector.platform.abort("test cleanup")
        collector.shutdown()


def test_pre_start_camera_and_pending_model_result_never_leak_into_new_session(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    pre_t = time.monotonic() - 1.0
    collector._last_camera_index = 10
    collector._last_result_index = 10
    collector._latest_tracking = {
        "frame_index": 10,
        "camera_host_t_s": pre_t,
        "camera_device_t_ms": 1000.0,
        "back_present": True,
        "initialized": True,
        "reason": "TRACKING",
        "points": [],
    }
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        boundary = collector._recording_boundary
        assert boundary["camera_start_boundary_index"] == 10
        assert boundary["model_start_boundary_index"] == 10

        # This frame was cloned before Start and only reached the lock afterward.
        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=10, t=pre_t, device_t_ms=1000.0,
        )) is True
        assert stub.submitted == []

        # RGB-D is retained, but pre-Start inference is explicitly unavailable.
        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=11, t=boundary["start_host_t_s"] + 0.01, device_t_ms=1033.0,
        )) is True
        assert stub.bundles[-1].tracking["tracking_association"] == "unavailable"
        assert stub.bundles[-1].tracking["source_tracking_frame_index"] is None

        collector._latest_tracking = {
            "frame_index": 11,
            "camera_host_t_s": boundary["start_host_t_s"] + 0.01,
            "camera_device_t_ms": 1033.0,
            "back_present": False,
            "initialized": False,
            "reason": "NO_POSTERIOR_PERSON",
            "points": [],
        }
        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=12, t=boundary["start_host_t_s"] + 0.02, device_t_ms=1066.0,
        )) is True
        assert stub.submitted == [11, 12]
        assert stub.bundles[-1].tracking["source_tracking_frame_index"] == 11
    finally:
        collector._abort_recording("test cleanup")
        collector.shutdown()


def test_end_persists_capture_model_boundaries_before_finalization(tmp_path):
    server = _load_collect_server()
    contexts = []

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def record_event(self, _payload):
            pass

        def write_frame(self, _frame, _tracking):
            pass

        def set_finalization_context(self, payload):
            contexts.append(dict(payload))

        def finalize(self):
            return {
                "status": "diagnostic_complete",
                "training_eligible": False,
                "capture_boundary": contexts[-1],
            }

        def abort(self, _reason):
            pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    # This unit test isolates the End/model-boundary handoff.  The real
    # completion artifact validator is covered separately with sealed files.
    collector.platform.completion_validator = None
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        start_t = collector._recording_boundary["start_host_t_s"]
        assert collector._submit_ingress(
            server.SimpleNamespace(index=4, t=start_t + 0.01),
            {"frame_index": 4},
        )
        collector._last_result_index = 3
        assert collector.control("end")["ok"] is True
        assert collector.platform.wait_for_terminal(timeout=1.0)["state"] == "COMPLETE"
        assert contexts[-1]["last_submitted_camera_index"] == 4
        assert contexts[-1]["last_completed_camera_index"] == 4
        assert contexts[-1]["last_completed_model_index"] == 3
        assert contexts[-1]["end_camera_frame_has_completed_model_result"] is False
        assert contexts[-1]["inference_timeline_policy"] == (
            "latest_only_recomputable_from_pinned_rgbd"
        )
        assert contexts[-1]["offline_model_recompute_from_pinned_rgbd"] is True
    finally:
        collector.shutdown()


def test_live_fault_after_end_boundary_cannot_abort_active_finalizer(tmp_path):
    server = _load_collect_server()
    entered = threading.Event()
    release = threading.Event()
    aborted = []

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def record_event(self, _payload):
            pass

        def write_frame(self, _frame, _tracking):
            pass

        def finalize(self):
            entered.set()
            assert release.wait(1.0)
            return {"status": "diagnostic_complete", "training_eligible": False}

        def abort(self, reason):
            aborted.append(reason)

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.platform.capture_factory = lambda _path: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector.platform.completion_validator = None
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector.control("end")["state"] == "FINALIZING"
        assert entered.wait(1.0)
        collector._abort_recording("camera became STALE")
        assert collector.platform.state.value == "FINALIZING"
        assert aborted == []
        release.set()
        assert collector.platform.wait_for_terminal(timeout=1.0)["state"] == "COMPLETE"
    finally:
        release.set()
        collector.shutdown()


def test_capture_fifo_overflow_aborts_immediately_and_never_seals(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        stub.submit = lambda _bundle: False
        stub.overflow = 1
        frame = server.SimpleNamespace(
            index=20,
            t=collector._recording_boundary["start_host_t_s"] + 0.01,
            device_t_ms=2000.0,
        )

        assert collector._submit_camera_capture(frame) is False

        assert collector.platform.state.value == "ABORTED"
        assert "overflow" in collector.platform.last_error.lower()
        assert not (collector.platform.session_dir / "DIAGNOSTIC_FINALIZATION.json").exists()
    finally:
        collector.shutdown()


def test_camera_index_gap_aborts_at_submission_boundary(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        start_t = collector._recording_boundary["start_host_t_s"]
        assert collector._submit_camera_capture(
            server.SimpleNamespace(index=30, t=start_t + 0.01, device_t_ms=3000.0),
        ) is True
        assert collector._submit_camera_capture(
            server.SimpleNamespace(index=32, t=start_t + 0.02, device_t_ms=3200.0),
        ) is False
        assert collector.platform.state.value == "ABORTED"
        assert "camera index gap" in collector.platform.last_error.lower()
    finally:
        collector.shutdown()


def test_first_post_start_camera_index_must_follow_start_boundary_exactly(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    collector._last_camera_index = 40
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=42,
            t=collector._recording_boundary["start_host_t_s"] + 0.01,
            device_t_ms=4200.0,
        )) is False
        assert collector.platform.state.value == "ABORTED"
        assert "first capture frame" in collector.platform.last_error.lower()
        assert stub.submitted == []
    finally:
        collector.shutdown()


def test_start_snapshots_camera_slot_before_persisting_strict_boundary(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    collector._last_camera_index = 40

    class Camera:
        @staticmethod
        def latest_frame():
            return server.SimpleNamespace(
                index=41, t=time.monotonic() - 0.001, status="OK",
                frame=np.zeros((2, 2, 3), dtype=np.uint8),
            )

        @staticmethod
        def stop():
            pass

    collector._cam = Camera()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector._recording_boundary["camera_start_boundary_index"] == 41
        boundary_t = collector._recording_boundary["start_host_t_s"]
        assert collector._submit_camera_capture(server.SimpleNamespace(
            index=42, t=boundary_t + 0.01, device_t_ms=4200.0,
        )) is True
    finally:
        collector._abort_recording("test cleanup")
        collector.shutdown()


def test_slow_capture_start_latches_camera_boundary_only_after_files_are_ready(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    ingress = _IngressStub()
    collector._ingress_worker = ingress
    boundary_contexts = []

    class Camera:
        def __init__(self):
            self.index = 40
            self.queue = []
            self.interval_active = False

        def _frame(self, index=None):
            value = self.index if index is None else int(index)
            return server.SimpleNamespace(
                index=value, t=time.monotonic(), status="OK",
                frame=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16), depth_scale=1.0,
                device_t_ms=float(value), time_source="test",
            )

        def latest_frame(self):
            return self._frame()

        def begin_delivery_interval(self, timeout_s=0):
            del timeout_s
            assert capture.started is True
            self.interval_active = True
            started = time.monotonic()
            return server.SimpleNamespace(
                t_start=started,
                frame=self._frame(),
                proof={
                    "boundary_latched_atomically": True,
                    "start_boundary_frame_index": self.index,
                    "start_boundary_host_t": started,
                },
            )

        def drain_delivery(self, max_frames=None):
            limit = len(self.queue) if max_frames is None else int(max_frames)
            batch, self.queue = self.queue[:limit], self.queue[limit:]
            return tuple(batch)

        def delivery_health(self):
            return {
                "status": "OK", "interval_active": self.interval_active,
                "accepting": self.interval_active, "queue_depth": len(self.queue),
                "interval_overflow_frames": 0, "overflow_frames": 0,
            }

        def abort_delivery_interval(self):
            self.interval_active = False
            self.queue.clear()
            return self.delivery_health()

        def stop(self):
            pass

    camera = Camera()

    class Capture:
        def request_cancel(self):
            pass

        started = False

        def start(self):
            # Simulate durable file initialization taking long enough for ten
            # camera frames to pass. The recording boundary must be sampled now,
            # not from the slot that existed before this work began.
            self.started = True
            camera.index = 50

        def set_recording_boundary(self, payload):
            boundary_contexts.append(dict(payload))

        def record_event(self, _payload):
            pass

        def write_frame(self, _frame, _tracking):
            pass

        def finalize(self):
            return {"training_eligible": False}

        def abort(self, _reason):
            pass

    capture = Capture()
    collector.platform.capture_factory = lambda _path: capture
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector._cam = camera
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector._recording_boundary["camera_start_boundary_index"] == 50
        assert boundary_contexts[-1]["camera_start_boundary_index"] == 50

        camera.queue.append(camera._frame(51))
        assert collector._drain_camera_delivery() == 1
        assert ingress.submitted == [51]
        assert collector.platform.state.value == "RECORDING"
    finally:
        collector._abort_recording("test cleanup")
        collector.shutdown()


def test_camera_delivery_overflow_aborts_recording_and_never_reports_complete(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()

    class Camera:
        overflow = 0

        @staticmethod
        def latest_frame():
            return server.SimpleNamespace(
                index=10, t=time.monotonic(), status="OK",
                frame=np.zeros((2, 2, 3), dtype=np.uint8),
            )

        @staticmethod
        def begin_delivery_interval(timeout_s=0):
            del timeout_s
            started = time.monotonic()
            return server.SimpleNamespace(
                t_start=started,
                frame=Camera.latest_frame(),
                proof={
                    "boundary_latched_atomically": True,
                    "start_boundary_frame_index": 10,
                    "start_boundary_host_t": started,
                },
            )

        @staticmethod
        def drain_delivery(max_frames=None):
            del max_frames
            return ()

        @classmethod
        def delivery_health(cls):
            return {
                "status": "OVERFLOW" if cls.overflow else "OK",
                "interval_active": True, "accepting": not bool(cls.overflow),
                "queue_depth": 0, "interval_overflow_frames": cls.overflow,
                "overflow_frames": cls.overflow,
                "first_overflow_frame_index": 11 if cls.overflow else None,
            }

        @staticmethod
        def abort_delivery_interval():
            return Camera.delivery_health()

        @staticmethod
        def stop():
            pass

    collector._cam = Camera()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        session_dir = collector.platform.session_dir
        Camera.overflow = 1

        assert collector._drain_camera_delivery() == 0
        assert collector.platform.state.value == "ABORTED"
        assert "delivery FIFO overflow" in collector.platform.last_error
        assert not (session_dir / "DIAGNOSTIC_FINALIZATION.json").exists()
    finally:
        collector.shutdown()


def test_model_starting_sentinel_is_not_treated_as_a_missing_camera_frame(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )

    class Worker:
        @staticmethod
        def latest():
            return {"frame_index": 0, "reason": "MODEL_STARTING", "points": []}

    collector._model_worker = Worker()
    try:
        assert collector._consume_model_result() is False
        assert collector._fatal_error is None
    finally:
        collector._model_worker = None
        collector.shutdown()


def test_model_completion_updates_tracking_timeline_but_never_submits_capture_frame(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    stub = _IngressStub()
    collector._ingress_worker = stub
    frame = server.SimpleNamespace(
        index=5, t=10.0, device_t_ms=500.0, age_s=0.0, status="OK",
    )

    class Worker:
        @staticmethod
        def latest():
            return {
                "frame_index": 5, "back_present": False,
                "initialized": False, "reason": "NO_POSTERIOR_PERSON", "points": [],
            }

        @staticmethod
        def health():
            return {"status": "OK", "error": None}

    collector._model_worker = Worker()
    collector._frame_ring[5] = frame
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector._consume_model_result() is True
        assert collector._latest_tracking["frame_index"] == 5
        assert stub.submitted == []
    finally:
        collector._model_worker = None
        collector.platform.abort("test cleanup")
        collector.shutdown()


def test_expired_model_source_frame_is_dropped_without_resetting_recording(tmp_path):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )

    class Worker:
        @staticmethod
        def latest():
            return {"frame_index": 99, "reason": "TRACKING", "points": []}

    collector._model_worker = Worker()
    try:
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        assert collector._consume_model_result() is True
        assert collector.platform.state.value == "RECORDING"
        assert collector._last_result_index == 99
        assert collector._model_status == "CATCHING_UP"
        assert collector._latest_tracking["reason"] == "MODEL_RESULT_EXPIRED"
    finally:
        collector._model_worker = None
        collector._abort_recording("test cleanup")
        collector.shutdown()


def _wait_until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class _LoopCamera:
    def __init__(self, *, stale_event=None):
        self.stale_event = stale_event
        self.index = 0
        self.stopped = False
        self.negotiated_mode = {
            "width": 64, "height": 48, "fps": 30.0,
            "color_format": "synthetic_rgb",
        }
        self.device_identity = "synthetic"

    def start(self):
        return self

    def latest_frame(self):
        self.index += 1
        stale = self.stale_event is not None and self.stale_event.is_set()
        return SimpleNamespace(
            frame=None if stale else np.zeros((48, 64, 3), dtype=np.uint8),
            depth=None if stale else np.full((48, 64), 900, dtype=np.uint16),
            depth_scale=1.0,
            t=time.monotonic(),
            index=self.index,
            status="STALE" if stale else "OK",
            age_s=1.0 if stale else 0.0,
            device_t_ms=float(self.index * 33),
            time_source="test",
        )

    def stop(self):
        self.stopped = True


def test_camera_open_failure_retries_with_backoff_and_recovers_without_busy_loop(tmp_path):
    server = _load_collect_server()
    attempts = []

    def factory(**_kwargs):
        attempts.append(time.monotonic())
        if len(attempts) == 1:
            raise RuntimeError("camera temporarily busy")
        return _LoopCamera()

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        camera_factory=factory,
        camera_reopen_interval_s=0.08,
    )
    try:
        assert _wait_until(
            lambda: collector._camera_status == "OK"
            and collector._model_status == "RUNNING"
            and collector.state_snapshot()["diagnostic_ready"] is True,
        )
        assert len(attempts) == 2
        assert attempts[1] - attempts[0] >= 0.06
        assert collector.state_snapshot()["diagnostic_ready"] is True
    finally:
        collector.shutdown()


def test_model_worker_error_clears_points_and_reopens_pipeline_with_backoff(
    tmp_path,
):
    server = _load_collect_server()
    attempts = []
    cameras = []
    processor_generations = []
    reset_evidence = []

    def camera_factory(**_kwargs):
        attempts.append(time.monotonic())
        camera = _LoopCamera()
        camera.device_identity = "CL8326300A0"
        cameras.append(camera)
        return camera

    def processor_factory(_collector, _frame):
        generation = len(processor_generations) + 1
        processor_generations.append(generation)
        if generation == 1:
            def fail(_model_frame):
                raise RuntimeError("simulated CUDA worker failure")
            return fail

        return lambda model_frame: {
            "frame_index": int(model_frame.index),
            "back_present": False,
            "initialized": False,
            "reason": "NO_PERSON",
            "points": [],
        }

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        commissioning_preview_only=True,
        camera_backend="orbbec",
        camera_id="CL8326300A0",
        camera_width=1920,
        camera_height=1080,
        camera_fps=30,
        camera_color_format="MJPG",
        depth_width=640,
        depth_height=576,
        depth_fps=30,
        depth_format="Y16",
        out_dir=tmp_path,
        checkpoint="/runtime/best.pt",
        checkpoint_sha256="a" * 64,
        base_model="/runtime/dinov2.pth",
        base_model_sha256="b" * 64,
        model_release_id="perc_lmh_v9_occ-a",
        posterior_gate_checkpoint="/runtime/yolo11x-pose.pt",
        posterior_gate_checkpoint_sha256="c" * 64,
        posterior_gate_release_id="yolo11x_pose-c",
        runtime_release_id="perception_runtime-d",
        runtime_lock="/runtime/lock.json",
        runtime_lock_sha256="d" * 64,
        code_release_id="code_release-e",
        ffmpeg_path="/runtime/ffmpeg",
        ffmpeg_sha256="f" * 64,
        require_force=False,
        require_robot=False,
        camera_factory=camera_factory,
        processor_factory=processor_factory,
        camera_reopen_interval_s=0.08,
        start_thread=False,
    )
    # 2026-08-11 operator policy: a model-worker fault must NOT touch the
    # camera (the old reset caused the seconds-on/seconds-off video cycle).
    # The worker is discarded, flagged, and rebuilt from the model cache.
    original_reset = collector._reset_camera_pipeline

    def observed_reset(reason):
        reset_evidence.append(str(reason))
        original_reset(reason)

    collector._reset_camera_pipeline = observed_reset
    collector.start()
    try:
        assert _wait_until(
            lambda: len(processor_generations) >= 2
            and collector._model_status == "RUNNING",
            timeout=2.0,
        )
        assert reset_evidence == []            # camera pipeline NEVER reset
        assert len(cameras) == 1               # same camera instance throughout
        assert cameras[0].stopped is False
        assert collector._camera_status == "OK"
        flags = list(collector._session_quality_flags)
        assert any(
            flag["kind"] == "model_worker_fault"
            and "simulated CUDA worker failure" in flag["detail"]
            for flag in flags
        )
    finally:
        collector.shutdown()


def test_stale_camera_aborts_recording_clears_points_and_reopens_pipeline(tmp_path):
    server = _load_collect_server()
    stale = threading.Event()
    cameras = []

    def factory(**_kwargs):
        camera = _LoopCamera(stale_event=stale if not cameras else None)
        cameras.append(camera)
        return camera

    class Capture:
        def request_cancel(self):
            pass

        def __init__(self):
            self.aborted = []

        def start(self):
            pass

        def record_event(self, _payload):
            pass

        def write_frame(self, _frame, _tracking):
            pass

        def finalize(self):
            return {"training_eligible": False}

        def abort(self, reason):
            self.aborted.append(reason)

    capture = Capture()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        camera_factory=factory,
        camera_reopen_interval_s=0.2,
    )
    collector.platform.capture_factory = lambda _path: capture
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    try:
        assert _wait_until(lambda: collector._model_status == "RUNNING")
        collector._camera_status = "OK"
        collector._model_status = "RUNNING"
        collector._runtime_health_snapshot = (
            lambda **_k: {"strict_ok": True, "blockers": []})
        assert collector.control("start")["ok"] is True
        stale.set()
        # sustained staleness passes the 2 s hysteresis and still resets
        assert _wait_until(
            lambda: collector._camera_status == "RECONNECTING", timeout=4.0
        )
        state = collector.state_snapshot()
        assert state["state"] == "ABORTED"
        assert state["points"] == []
        assert state["diagnostic_ready"] is False
        assert capture.aborted and "STALE" in capture.aborted[-1]
        assert not (collector.platform.session_dir / "DIAGNOSTIC_FINALIZATION.json").exists()
        assert _wait_until(
            lambda: len(cameras) >= 2
            and collector._camera_status == "OK"
            and collector._model_status == "RUNNING",
            timeout=2.0,
        )
        assert cameras[0].stopped is True
    finally:
        collector.shutdown()


def test_served_start_authorized_folds_in_the_audio_gate(tmp_path):
    # The page consumes only snapshot.start_authorized; a missing stethoscope
    # signal must never leave it true (the reviewer's page/API mismatch).
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False, require_audio=True,
    )
    try:
        collector._audio_snapshot = lambda: {
            "status": "OK", "signal_present": False,
        }
        silent = collector.state_snapshot()
        assert silent["start_authorized"] is False
        assert silent["start_rejection"]
    finally:
        collector.shutdown()


def test_control_start_and_page_agree_when_device_is_not_ready(tmp_path):
    # The reviewer's repro: with camera/model not running, the page shows
    # start_authorized false AND the control path must refuse too (no empty
    # recording).  They consult the same device-readiness precondition.
    server = _load_collect_server()

    class Capture:
        def request_cancel(self):
            pass

        def start(self): pass
        def record_event(self, _p): pass
        def write_frame(self, _f, _t): pass
        def finalize(self): return {"status": "diagnostic_complete",
                                     "training_eligible": False}
        def abort(self, _r): pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.platform.capture_factory = lambda _p: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    try:
        # Device threads never started -> camera not OK, model not RUNNING.
        assert collector.state_snapshot()["start_authorized"] is False
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "start_not_authorized"
        assert collector.platform.state.value != "RECORDING"
    finally:
        collector.shutdown()


def test_start_is_refused_when_a_worker_is_dead_despite_cached_status(tmp_path):
    # The reviewer's repro: cached camera_status/model_status still say OK/RUNNING
    # but a worker actually died.  Start must consult live runtime health (the
    # same strict gate Confirm uses), not the stale cache, and the page must
    # agree.
    server = _load_collect_server()

    class Capture:
        def request_cancel(self):
            pass

        def start(self): pass
        def record_event(self, _p): pass
        def write_frame(self, _f, _t): pass
        def finalize(self): return {"status": "diagnostic_complete",
                                     "training_eligible": False}
        def abort(self, _r): pass

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector.platform.capture_factory = lambda _p: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    try:
        collector._camera_status = "OK"       # stale cache
        collector._model_status = "RUNNING"   # stale cache
        collector._runtime_health_snapshot = (
            lambda **_k: {
                "strict_ok": False,
                "blockers": ["model worker is not alive and OK"],
            })

        assert collector.state_snapshot()["start_authorized"] is False
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "start_not_authorized"
        assert "model worker" in result["message"]
        assert collector.platform.state.value != "RECORDING"
    finally:
        collector.shutdown()


def test_start_rolls_back_when_a_worker_dies_during_the_arm(tmp_path):
    # A worker that dies during capture.start()/sensor init (after the entry
    # re-check) must not leave RECORDING: the health epoch advanced, so the final
    # arm arbitration rolls the whole Start back.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            # Simulate a worker dying mid-arm: the loop would have bumped this.
            collector._health_generation += 1

        def record_event(self, _p): pass
        def write_frame(self, _f, _t): pass
        def finalize(self): return {"status": "diagnostic_complete",
                                    "training_eligible": False}
        def abort(self, _r): pass

    collector.platform.capture_factory = lambda _p: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    try:
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "runtime_degraded_during_start"
        assert collector.platform.state.value != "RECORDING"
    finally:
        collector.shutdown()


def test_start_rolls_back_when_a_worker_dies_right_before_ingress_opens(tmp_path):
    # Closes the residual window between the arm-time health check (before the
    # camera boundary latch) and the ingress open: a death detected only at the
    # FINAL arm_start re-evaluation must still roll the whole Start back rather
    # than open capture ingress on a dead model.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )

    class Capture:
        def request_cancel(self): pass
        def start(self): pass  # healthy at capture.start()
        def record_event(self, _p): pass
        def write_frame(self, _f, _t): pass
        def finalize(self): return {"status": "diagnostic_complete",
                                    "training_eligible": False}
        def abort(self, _r): pass

    collector.platform.capture_factory = lambda _p: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    # Every earlier liveness gate passes; the worker death is observable ONLY as a
    # health-epoch advance inside the final atomic arm (arm_interval_if_healthy):
    # its fresh degraded snapshot advances the epoch under the epoch lock, so the
    # arm sees epoch != expected and refuses to open capture ingress.  The whole
    # Start must roll back rather than record on a dead model.
    collector._device_start_rejection = lambda **_k: None
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": False, "blockers": ["worker died before ingress open"]})

    try:
        result = collector.control("start")
        assert result["ok"] is False
        assert collector.platform.state.value != "RECORDING"
        # Ingress must remain paused (never opened on the dead model).
        assert collector._capture_ingress_paused is True
    finally:
        collector.shutdown()


class _ArmHealth:
    def __init__(self, status, preflight_ready=False):
        self.status = status
        self.preflight_ready = preflight_ready
        self.error = ""


class _ArmCoord:
    def __init__(self, script):
        self._script = list(script)
        self.events = []
        self._faulted = False
        self._fr = ""

    def renew_receiver(self):
        self.events.append("renew")
        self._faulted = False
        self._fr = ""

    def prepare_receiver_socket(self): self.events.append("prepare")
    def attach_controller_process(self, a): self.events.append("attach")
    def start_receiver(self): self.events.append("start")
    def stop_receiver(self): self.events.append("stop")

    def receiver_health(self, **k):
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]

    @property
    def faulted(self): return self._faulted

    @property
    def fault_reason(self): return self._fr


class _ArmCtrl:
    def __init__(self): self._exit = None; self.terminated = False
    @property
    def pid(self): return 999
    def poll(self): return self._exit
    def terminate(self): self.terminated = True


class _ArmDisable:
    def __init__(self): self.calls = []
    def __call__(self, reason):
        self.calls.append(reason)
        return type("O", (), {"outcome": "energy_isolated"})()


def _configure_arm(collector, coord, disable):
    # Production cannot obtain this capability.  It is accepted only by a fake
    # Collector and keeps all legacy lifecycle tests synthetic.
    collector._install_test_only_fake_arm_capability()
    collector.require_robot = True
    collector.commissioning_preview_only = False
    collector._robot_coordinator = coord
    collector._controller_launcher = lambda: _ArmCtrl()
    collector._controller_disable = disable
    collector._controller_attest = lambda pid, uid: f"attest{pid}"
    collector._controller_verify_motor_disable = lambda: True
    collector._controller_readiness_timeout_s = 2.0


def test_arm_authority_rejects_all_pre_hil_entry_points_without_callbacks(tmp_path):
    server = _load_collect_server()
    calls = {name: 0 for name in ("launch", "disable", "preflight", "attest", "verify")}

    def counted(name, result=None):
        def callback(*_args, **_kwargs):
            calls[name] += 1
            return result
        return callback

    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=_ArmCoord([_ArmHealth("IDLE")]),
        controller_launcher=counted("launch", _ArmCtrl()),
        controller_disable=counted("disable"),
        controller_preflight=counted("preflight"),
        controller_attest=counted("attest", "attestation"),
        controller_verify_motor_disable=counted("verify", False),
        controller_launch_release_id="controller_launch_v1-" + "1" * 64,
        controller_launch_contract_sha256="1" * 64,
        controller_open_target_status="HIL_PENDING",
        start_thread=False,
    )
    try:
        with pytest.raises(RuntimeError, match="only in fake mode"):
            c._install_test_only_fake_arm_capability()
        authority = c.arm_control_authorization()
        assert authority.arm_bringup_authorized is False
        assert authority.arm_stop_authorized is False
        assert authority.arm_bringup_rejection == "Safe-stop HIL required"
        assert authority.arm_stop_rejection == "Safe-stop HIL required"
        with pytest.raises((AttributeError, TypeError)):
            authority.arm_bringup_authorized = True

        state = c.state_snapshot()
        controller = state["controller"]
        assert controller["bringup_active"] is False
        assert controller["arm_bringup_authorized"] is False
        assert controller["arm_stop_authorized"] is False
        assert controller["arm_bringup_rejection"] == "Safe-stop HIL required"
        assert controller["arm_stop_rejection"] == "Safe-stop HIL required"

        results = (
            c.bring_up_controller(),
            c.disable_controller(),
            c.control("arm_bringup"),
            c.control("arm_disable"),
            c.control(" ARM_DISABLE "),
        )
        assert all(result["ok"] is False for result in results)
        assert {result["message"] for result in results} == {
            "Safe-stop HIL required"
        }
        assert calls == {name: 0 for name in calls}
        assert c._controller_companion is None
    finally:
        c.shutdown()
    assert calls == {name: 0 for name in calls}


def test_raw_http_arm_actions_match_state_rejection_and_call_nothing(tmp_path):
    server = _load_collect_server()
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("pre-HIL callback must not execute")

    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=_ArmCoord([_ArmHealth("IDLE")]),
        controller_launcher=forbidden,
        controller_disable=forbidden,
        controller_preflight=forbidden,
        controller_attest=forbidden,
        controller_verify_motor_disable=forbidden,
        start_thread=False,
    )
    state_reason = c.state_snapshot()["controller"]["arm_bringup_rejection"]
    handler_type = server.make_handler(c, b"page")
    try:
        for action in ("arm_bringup", "arm_disable", " ARM_DISABLE "):
            raw = json.dumps({"action": action}).encode()
            handler = object.__new__(handler_type)
            handler.path = "/control"
            handler.headers = {"Content-Length": str(len(raw))}
            handler.rfile = io.BytesIO(raw)
            sent = {}

            def capture_send(body, content_type, *, status=200, no_store=False):
                sent.update(
                    body=body,
                    content_type=content_type,
                    status=status,
                    no_store=no_store,
                )

            handler._send = capture_send
            handler.do_POST()
            result = json.loads(sent["body"])
            assert sent["status"] == 409
            assert result["ok"] is False
            assert result["message"] == state_reason == "Safe-stop HIL required"
        assert calls == []
    finally:
        c.shutdown()


def test_dormant_launch_metadata_is_visible_but_never_authorizes_arm(tmp_path):
    server = _load_collect_server()
    release_id = "controller_launch_v1-" + "2" * 64
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=False,
        out_dir=tmp_path,
        require_robot=False,
        controller_launch_release_id=release_id,
        controller_launch_contract_sha256="2" * 64,
        controller_open_target_status="HIL_PENDING",
        start_thread=False,
    )
    try:
        payload = c._controller_status_payload()
        assert payload["visible"] is True
        assert payload["configured"] is False
        assert payload["launch_release_id"] == release_id
        assert payload["launch_contract_sha256"] == "2" * 64
        assert payload["open_target_status"] == "HIL_PENDING"
        assert payload["arm_bringup_authorized"] is False
        assert payload["arm_stop_authorized"] is False
        assert payload["arm_bringup_rejection"] == "Safe-stop HIL required"
    finally:
        c.shutdown()


def test_nonfake_cli_rejects_all_raw_controller_commands(capsys):
    server = _load_collect_server()
    for argv in (
        ["--teleop-launch-cmd", "/tmp/controller"],
        ["--controller-disable-cmd", "/tmp/disable"],
        ["--controller-preflight-cmd", "/tmp/preflight"],
    ):
        with pytest.raises(SystemExit):
            server.main(argv)
        assert "raw controller command flags are forbidden in real mode" in capsys.readouterr().err


def test_bringup_rechecks_arm_authority_before_companion_creation(tmp_path):
    server = _load_collect_server()
    launch_calls = []

    def launcher():
        launch_calls.append("launch")
        return _ArmCtrl()

    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    c._controller_launcher = launcher
    derive = c.arm_control_authorization
    derivations = 0

    def expiring_authority():
        nonlocal derivations
        derivations += 1
        if derivations == 2:
            c._arm_control_capability = None
        return derive()

    c.arm_control_authorization = expiring_authority
    try:
        result = c.bring_up_controller()
        assert result["ok"] is False
        assert result["message"] == "Safe-stop HIL required"
        assert derivations == 2
        assert launch_calls == []
        assert c._controller_companion is None
    finally:
        c.shutdown()


def test_normalized_arm_disable_control_bypasses_control_lock(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    disable = _ArmDisable()
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), disable)
    result = {}
    done = threading.Event()
    worker = None
    try:
        assert c.bring_up_controller()["ok"] is True

        def stop_from_another_request():
            result.update(c.control(" ARM_DISABLE "))
            done.set()

        c._control_lock.acquire()
        try:
            worker = threading.Thread(target=stop_from_another_request)
            worker.start()
            completed_while_locked = done.wait(0.25)
        finally:
            c._control_lock.release()
        worker.join(timeout=1.0)

        assert completed_while_locked is True
        assert worker.is_alive() is False
        assert result["ok"] is True
        assert result["code"] == "controller_disabled"
        assert disable.calls == ["operator_estop"]
    finally:
        if worker is not None:
            worker.join(timeout=1.0)
        c.shutdown()


def test_authorized_idle_stop_runs_independent_disable_and_verifier(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    events = []

    def disable(reason):
        events.append(("disable", reason))
        return "disable_sent"

    def verify():
        events.append(("verify", None))
        return True

    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), disable)
    c._controller_verify_motor_disable = verify
    try:
        result = c.disable_controller("idle_operator_stop")
        assert result["ok"] is True
        assert result["code"] == "controller_disabled"
        assert result["energy_isolated"] is True
        assert result["motor_disable_verified"] is True
        assert events == [
            ("disable", "idle_operator_stop"),
            ("verify", None),
        ]
        assert c._controller_companion is None
    finally:
        # Remove the accepted capability so cleanup does not intentionally issue
        # a second independent idle-stop command.
        c._arm_control_capability = None
        c.shutdown()


def test_fake_hil_capability_without_motor_verifier_authorizes_nothing(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    disable = _ArmDisable()
    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), disable)
    c._controller_verify_motor_disable = None
    assert c._controller_verify_motor_disable is None
    try:
        authority = c.arm_control_authorization()
        assert authority.arm_bringup_authorized is False
        assert authority.arm_stop_authorized is False
        result = c.disable_controller("idle_operator_stop")
        assert result["ok"] is False
        assert result["message"] == "Safe-stop HIL required"
        assert disable.calls == []
    finally:
        c.shutdown()
    assert disable.calls == []


def test_shutdown_stops_authorized_idle_arm_before_any_teardown(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    events = []

    def disable(reason):
        events.append(("disable", reason))
        return "disable_sent"

    def verify():
        events.append(("verify", None))
        return True

    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), disable)
    c._controller_verify_motor_disable = verify
    clear_back = c._clear_back_confirmation

    def observed_teardown(reason):
        events.append(("teardown", reason))
        return clear_back(reason)

    c._clear_back_confirmation = observed_teardown
    c.shutdown()

    assert events[:3] == [
        ("disable", "service_shutdown"),
        ("verify", None),
        ("teardown", "SERVICE_SHUTDOWN"),
    ]


def test_shutdown_disables_before_a_blocked_recording_abort(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    events = []
    abort_entered = threading.Event()
    release_abort = threading.Event()

    def disable(reason):
        events.append(("disable", reason))
        return "disable_sent"

    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), disable)
    c._controller_verify_motor_disable = lambda: (
        events.append(("verify", None)) or True
    )
    c.platform.state = server.CollectionState.RECORDING

    def blocked_abort(reason):
        events.append(("abort", reason))
        abort_entered.set()
        assert release_abort.wait(2.0)

    c._abort_recording = blocked_abort
    worker = threading.Thread(target=c.shutdown)
    try:
        worker.start()
        assert abort_entered.wait(1.0)
        assert events[:3] == [
            ("disable", "service_shutdown"),
            ("verify", None),
            ("abort", "service shutdown before End"),
        ]
    finally:
        release_abort.set()
    worker.join(timeout=2.0)
    assert worker.is_alive() is False


def test_shutdown_continues_teardown_when_recording_abort_raises(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    events = []

    def disable(reason):
        events.append(("disable", reason))
        return "disable_sent"

    class Camera:
        def stop(self):
            events.append(("camera_stop", None))

    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), disable)
    c._controller_verify_motor_disable = lambda: (
        events.append(("verify", None)) or True
    )
    c.platform.state = server.CollectionState.RECORDING
    c._cam = Camera()

    def raising_abort(reason):
        events.append(("abort", reason))
        raise RuntimeError("synthetic abort failure")

    c._abort_recording = raising_abort
    c.shutdown()

    assert events[:3] == [
        ("disable", "service_shutdown"),
        ("verify", None),
        ("abort", "service shutdown before End"),
    ]
    assert ("camera_stop", None) in events


def test_same_page_stop_during_preflight_prevents_controller_launch(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    launch_calls = []

    def preflight():
        preflight_entered.set()
        assert release_preflight.wait(2.0)

    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    c._controller_preflight = preflight
    c._controller_launcher = lambda: (
        launch_calls.append("launch"), _ArmCtrl(),
    )[1]
    c._controller_verify_motor_disable = lambda: True
    bringup_result = {}
    worker = threading.Thread(
        target=lambda: bringup_result.update(c.control("arm_bringup")),
    )
    try:
        worker.start()
        assert preflight_entered.wait(1.0)
        stop_result = c.control(" ARM_DISABLE ")
        assert stop_result["ok"] is True
        assert stop_result["energy_isolated"] is True
    finally:
        release_preflight.set()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert bringup_result["ok"] is False
    assert launch_calls == []
    assert c._controller_companion.stop_requested is True
    c.shutdown()


def test_estop_during_blocking_launch_stays_pending_until_fresh_disable(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    launch_entered = threading.Event()
    release_launch = threading.Event()
    controller = _ArmCtrl()
    disable_reasons = []
    verify_calls = []

    def launcher():
        launch_entered.set()
        assert release_launch.wait(2.0)
        return controller

    def disable(reason):
        disable_reasons.append(reason)
        return "disable_sent"

    def verify():
        verify_calls.append("verified")
        return True

    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), disable)
    c._controller_launcher = launcher
    c._controller_verify_motor_disable = verify
    bringup_result = {}
    worker = threading.Thread(
        target=lambda: bringup_result.update(c.control("arm_bringup")),
    )
    try:
        worker.start()
        assert launch_entered.wait(1.0)

        immediate = c.control("arm_disable")
        assert immediate["ok"] is True
        assert immediate["code"] == "controller_disable_pending"
        assert immediate["isolation_pending"] is True
        assert immediate["energy_isolated"] is False
        assert immediate["motor_disable_verified"] is False
        pending = c._controller_status_payload()
        assert pending["isolation_pending"] is True
        assert pending["energy_isolated"] is False

        release_launch.set()
        worker.join(timeout=2.0)
        assert worker.is_alive() is False
        assert bringup_result["ok"] is False
        assert controller.terminated is True
        assert disable_reasons == [
            "operator_estop",
            "late controller terminated during E-stop",
        ]
        assert verify_calls == ["verified", "verified"]
        final = c._controller_status_payload()
        assert final["isolation_pending"] is False
        assert final["energy_isolated"] is True
        assert final["motor_disable_verified"] is True
    finally:
        release_launch.set()
        worker.join(timeout=2.0)
        c.shutdown()


def test_production_teleop_requires_real_hardware(tmp_path):
    # #6 step C: production teleop capture is incompatible with fake/preview data.
    server = _load_collect_server()
    with pytest.raises(ValueError, match="production teleop"):
        server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, production_teleop=True,
                         start_thread=False)


def test_capture_factory_selects_writer_by_mode(tmp_path):
    # #6 step C: the capture factory builds ProductionTeleopCapture in production mode,
    # DiagnosticPerceptionCapture otherwise (and production IS a diagnostic subclass, so
    # it reuses all the machinery).
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    (tmp_path / "s1").mkdir()
    cap = c._capture_factory(tmp_path / "s1")
    assert type(cap).__name__ == "DiagnosticPerceptionCapture"
    c.production_teleop = True  # flip mode (bypassing the ctor guard) and re-check
    (tmp_path / "s2").mkdir()
    cap2 = c._capture_factory(tmp_path / "s2")
    assert type(cap2).__name__ == "ProductionTeleopCapture"
    assert isinstance(cap2, type(cap))


def test_finite_json_sanitizes_infinity_and_nan_for_state():
    # /state must be strict JSON the browser can parse: non-finite floats (e.g.
    # tracking_age_s before the first tracking result) become null.
    server = _load_collect_server()
    raw = {
        "tracking_age_s": math.inf,
        "nested": {"x": math.nan, "ok": 1.5},
        "list": [math.inf, 2.0],
        "s": "keep",
    }
    cleaned = server._finite_json(raw)
    assert cleaned["tracking_age_s"] is None
    assert cleaned["nested"]["x"] is None
    assert cleaned["nested"]["ok"] == 1.5
    assert cleaned["list"] == [None, 2.0]
    assert cleaned["s"] == "keep"

    def _reject(_token):
        raise ValueError("non-finite token in served /state")

    json.loads(json.dumps(cleaned), parse_constant=_reject)  # browser-equivalent parse


def test_arm_bringup_ready_then_disable(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    try:
        assert c._controller_configured() is True
        assert c._controller_status_payload()["state"] == "idle"
        r = c.bring_up_controller()
        assert r["ok"] and r["code"] == "controller_ready" and r["controller_pid"] == 999
        assert c._controller_ready() is True
        # renew_receiver runs first: every bring-up gets a fresh single-use receiver
        assert coord.events[:4] == ["renew", "prepare", "attach", "start"]
        assert disable.calls == []                       # never disabled on success
        d = c.disable_controller()
        assert d["ok"] and d["code"] == "controller_disabled"
        assert disable.calls == ["operator_estop"]
        assert c._controller_ready() is False
    finally:
        c.shutdown()


def test_failed_motor_readback_is_not_energy_isolated(tmp_path):
    # A verifier that explicitly fails must surface command-succeeded/UNVERIFIED,
    # never "energy-isolated".
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])
    _configure_arm(c, coord, _ArmDisable())
    c._controller_verify_motor_disable = lambda: False
    try:
        c.bring_up_controller()
        d = c.disable_controller()
        assert d["ok"] and d["code"] == "controller_disabled"
        assert d["energy_isolated"] is False
        assert d["motor_disable_verified"] is False
        assert "UNVERIFIED" in d["message"]
        p = c._controller_status_payload()
        assert p["disable_command_succeeded"] is True
        assert p["energy_isolated"] is False
        assert p["motor_disable_verified"] is False
    finally:
        c.shutdown()


def test_arm_preflight_runs_before_launch(tmp_path):
    # P0-1: a configured controller preflight (clear_faults + socket dir) runs
    # synchronously BEFORE the controller is launched, so the launched process is the
    # controller binary and its exe attestation is sound.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])
    _configure_arm(c, coord, _ArmDisable())
    order = []
    c._controller_launcher = lambda: (order.append("launch"), _ArmCtrl())[1]
    c._controller_preflight = lambda: order.append("preflight")
    try:
        r = c.bring_up_controller()
        assert r["ok"] and r["code"] == "controller_ready"
        assert order == ["preflight", "launch"]
    finally:
        c.shutdown()


def test_arm_bringup_fault_energy_isolates(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("FAULT")])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    c._controller_verify_motor_disable = lambda: False
    try:
        r = c.bring_up_controller()
        assert r["ok"] is False and r["code"] == "controller_bring_up_faulted"
        assert len(disable.calls) == 1                   # disable commanded on failure
        assert r["energy_isolated"] is False             # explicit readback failure
        assert c._controller_ready() is False
    finally:
        c.shutdown()


def test_disable_command_failure_reports_not_ok(tmp_path):
    # Safety: if the disable COMMAND fails, the E-stop reports ok:false (arm NOT
    # confirmed disabled), never success.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])

    def raising_disable(reason):
        raise RuntimeError("CAN disable send failed")

    _configure_arm(c, coord, raising_disable)
    try:
        c.bring_up_controller()
        d = c.disable_controller()
        assert d["ok"] is False
        assert d["code"] == "controller_disable_command_failed"
        assert d["energy_isolated"] is False
    finally:
        c.shutdown()


def test_collector_loop_crash_does_not_cut_arm_power(tmp_path):
    # OPERATOR SAFETY POLICY: software never cuts arm power on its own -- not even
    # on a collector crash. The human-teleoperated arm stays powered; the fatal
    # error latches (bring-ups refused) and the operator decides via their three
    # channels (console action / physical E-stop / 8801 Disable).
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    try:
        c.bring_up_controller()
        assert c._controller_ready() is True

        def _boom():
            raise RuntimeError("synthetic loop crash")

        c._loop = _boom
        c._abort_recording = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("abort boom"))
        c._loop_guarded()
        assert not disable.calls                   # power was NOT cut
        assert c._controller_companion.disabled is False
        assert c._fatal_error                      # but the fatal latched
    finally:
        c.shutdown()


def test_bring_up_refused_when_collector_is_fatal(tmp_path):
    # Safety: a fatal collector loop (arm supervisor gone) refuses a fresh bring-up.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    try:
        c._fatal_error = "camera open failed"
        r = c.bring_up_controller()
        assert r["ok"] is False
        assert r["code"] == "controller_bring_up_refused_fatal"
    finally:
        c.shutdown()


def test_shutdown_between_outer_check_and_controller_lock_prevents_bringup(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    launch_calls = []
    c._controller_launcher = lambda: (launch_calls.append("launch"), _ArmCtrl())[1]
    first_authority_derived = threading.Event()
    derive = c.arm_control_authorization

    def observed_authority():
        authority = derive()
        first_authority_derived.set()
        return authority

    c.arm_control_authorization = observed_authority
    bringup_result = {}
    bringup = threading.Thread(
        target=lambda: bringup_result.update(c.bring_up_controller()),
    )
    shutdown = threading.Thread(target=c.shutdown)

    c._controller_lock.acquire()
    try:
        bringup.start()
        assert first_authority_derived.wait(1.0)
        shutdown.start()
        assert c._stop.wait(1.0)
    finally:
        c._controller_lock.release()
    bringup.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert bringup.is_alive() is False
    assert shutdown.is_alive() is False
    assert bringup_result["ok"] is False
    assert bringup_result["code"] == "controller_bring_up_refused_fatal"
    assert launch_calls == []
    assert c._controller_companion is None


def test_shutdown_observes_and_stops_companion_assigned_first(
    tmp_path, monkeypatch,
):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )

    class ObservedDisable(_ArmDisable):
        def __init__(self):
            super().__init__()
            self.called = threading.Event()

        def __call__(self, reason):
            outcome = super().__call__(reason)
            self.called.set()
            return outcome

    disable = ObservedDisable()
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), disable)
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    companion_type = server.TrustedControllerCompanion

    def blocking_constructor(**kwargs):
        constructor_entered.set()
        assert release_constructor.wait(2.0)
        return companion_type(**kwargs)

    monkeypatch.setattr(server, "TrustedControllerCompanion", blocking_constructor)

    def blocking_preflight():
        preflight_entered.set()
        assert release_preflight.wait(2.0)

    c._controller_preflight = blocking_preflight
    bringup_result = {}
    bringup = threading.Thread(
        target=lambda: bringup_result.update(c.bring_up_controller()),
    )
    shutdown = threading.Thread(target=c.shutdown)
    try:
        bringup.start()
        # Constructor is inside _controller_lock.  Shutdown sets _stop and then
        # waits on that lock; once construction is released, assignment happens
        # before shutdown can observe the companion.
        assert constructor_entered.wait(1.0)
        shutdown.start()
        assert c._stop.wait(1.0)
        release_constructor.set()
        assert preflight_entered.wait(1.0)
        assert disable.called.wait(1.0)
    finally:
        release_constructor.set()
        release_preflight.set()
    bringup.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert bringup.is_alive() is False
    assert shutdown.is_alive() is False
    assert bringup_result["ok"] is False
    assert c._controller_companion.disabled is True
    assert disable.calls[0] == "service_shutdown"


def test_watchdog_thread_latches_telemetry_alarm_without_power_cut(tmp_path):
    # OPERATOR SAFETY POLICY: the independent watchdog still supervises, but a
    # telemetry fault latches the LOUD alarm (data unusable) instead of cutting
    # power; the operator's three channels are the only power-off paths.
    import threading as _threading
    import time as _time
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    c._controller_watchdog_interval_s = 0.005
    try:
        c.bring_up_controller()
        assert c._controller_ready() is True
        wd = _threading.Thread(target=c._controller_watchdog_run, daemon=True)
        wd.start()
        coord._script = [_ArmHealth("STALE", False)]     # telemetry goes stale
        deadline = _time.monotonic() + 2.0
        comp = c._controller_companion
        while not comp.telemetry_fault_reason and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert "ARM STILL POWERED" in comp.telemetry_fault_reason
        assert comp.disabled is False                    # power NOT cut
        assert not disable.calls
    finally:
        c._stop.set()
        wd.join(timeout=1.0)
        c.shutdown()


def test_http_estop_bypasses_control_path_during_bringup():
    # HIGHEST-priority safety: an E-stop (arm_disable) POST must NOT queue behind an
    # in-flight bring-up/preflight/readiness-wait that holds the control path. It routes
    # straight to disable_controller, so it disarms the arm promptly even while control()
    # is blocked. If it still routed through control(), this request would hang.
    server = _load_collect_server()
    blocked = threading.Event()          # never set: simulates bring-up holding _control_lock
    disable_calls = []

    class Col:
        jpeg = b"jpeg"

        def control(self, action):
            blocked.wait(timeout=5.0)    # would hang the request if E-stop routed here
            return {"ok": True}

        def disable_controller(self):
            disable_calls.append("estop")
            return {
                "ok": True, "code": "controller_disabled",
                "message": "left arm energy-isolated",
            }

        def state_snapshot(self):
            return {}

    httpd = server.ThreadingHTTPServer(
        ("127.0.0.1", 0), server.make_handler(Col(), b"page"),
    )
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2.0)
    try:
        started = time.monotonic()
        body = json.dumps({"action": "arm_disable"})
        connection.request(
            "POST", "/control", body,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        elapsed = time.monotonic() - started
        assert response.status == 200
        assert result["code"] == "controller_disabled"
        assert disable_calls == ["estop"]     # went straight to disable_controller
        assert elapsed < 1.0                  # did NOT block on the held control path
    finally:
        blocked.set()
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2.0)


def test_operator_disable_failure_is_reported_and_retryable(tmp_path):
    # OPERATOR SAFETY POLICY: disables originate only from the operator. If the
    # command FAILS, the result says so honestly (ok=False, arm NOT confirmed
    # disabled) and the operator's retry succeeds -- no automatic re-issuing.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])
    attempts = {"n": 0}

    class _FlakyDisable:
        def __init__(self): self.calls = []
        def __call__(self, reason):
            self.calls.append(reason)
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("first disable send failed")   # command fails
            return type("O", (), {"outcome": "energy_isolated"})()

    disable = _FlakyDisable()
    _configure_arm(c, coord, disable)
    try:
        c.bring_up_controller()
        assert c._controller_ready() is True
        first = c.disable_controller()                    # operator presses Disable
        assert first["ok"] is False                       # honest failure report
        assert c._controller_companion.disabled is False
        second = c.disable_controller()                   # operator retries
        assert second["ok"] is True
        assert c._controller_companion.disabled is True
        assert attempts["n"] == 2                         # exactly the two operator actions
    finally:
        c.shutdown()


def test_bring_up_refuses_replacing_not_isolated_companion(tmp_path):
    # Safety: a fresh bring-up must NOT replace a companion whose controller was launched
    # but is not confirmed disabled (its disable command failed; the watchdog is
    # retrying). Replacing it would orphan a possibly-live controller (unreachable by the
    # watchdog AND the E-stop) and launch a second controller onto a non-isolated arm.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])

    class _FailDisable:
        def __init__(self): self.calls = []
        def __call__(self, reason):
            self.calls.append(reason)
            raise RuntimeError("disable send failed")   # never confirms

    _configure_arm(c, coord, _FailDisable())
    try:
        c.bring_up_controller()                          # arm comes up ready
        with pytest.raises(Exception):                   # fault -> disable attempt FAILS
            c._controller_companion.report_external_fault("telemetry fault")
        comp1 = c._controller_companion
        assert comp1.state == "faulted"
        assert comp1.disabled is False                   # not confirmed isolated
        assert comp1.controller_pid is not None          # a controller WAS launched
        r = c.bring_up_controller()                      # second bring-up must refuse
        assert r["ok"] is False
        assert r["code"] == "controller_prior_not_isolated"
        assert c._controller_companion is comp1          # companion NOT replaced/orphaned
    finally:
        c.shutdown()


def test_watchdog_survives_base_exception_in_tick(tmp_path):
    # The watchdog is the SOLE controller supervisor. A BaseException escaping poll() must
    # NOT kill the thread and silently leave the arm unsupervised: the outer guard
    # fail-safe-disables and the loop keeps running.
    import threading as _threading
    import time as _time
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    c._controller_watchdog_interval_s = 0.005
    wd = None
    try:
        c.bring_up_controller()

        def boom():
            raise BaseException("unexpected supervisor error")

        c._controller_companion.poll = boom              # escapes the inner Exception guard
        wd = _threading.Thread(target=c._controller_watchdog_run, daemon=True)
        wd.start()
        deadline = _time.monotonic() + 2.0
        while not c._controller_companion.disabled and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert c._controller_companion.disabled is True  # fail-safe disabled despite BaseException
        assert wd.is_alive()                             # supervisor survived, not killed
    finally:
        c._stop.set()
        if wd is not None:
            wd.join(timeout=1.0)
        c.shutdown()


def test_force_snapshot_never_reports_training_eligible(tmp_path):
    # Force is advisory-only: _force_snapshot must NEVER emit training_eligible=true, for
    # ANY force status -- including a valid over-threshold physician reading. Only the
    # finalizer promotes; no producer surface may assert eligibility from a force reading.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    c.require_force = True

    class _FakeForce:
        def __init__(self, status): self._status = status
        def read(self):
            return server.SimpleNamespace(
                status=self._status, force_n=3.0, age_s=0.0, displacement_mm=0.0,
                raw=0, hall_v=0.0, device_t_ms=None,
            )

    try:
        for status in ("OK", "WARN", "SCREENED_LIMIT", "HARD_LIMIT"):
            c._force = _FakeForce(status)
            snap = c._force_snapshot()
            assert snap["training_eligible"] is False, status
    finally:
        c.shutdown()


def test_state_exposes_anatomical_site_instructions(tmp_path):
    # The platform surfaces the VLA "L" grounding (each point code -> anatomical instruction)
    # so the operator sees the goal and the collection binds to the SAME shared source of
    # truth as the converter, not an opaque code.
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2)),
         server.SimpleNamespace(name="L5", uv=(0.4, 0.6))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        snap = c.state_snapshot()
        si = snap["site_instructions"]
        assert si["R1"] == "auscultate the right upper posterior lung field"
        assert si["L5"] == "auscultate the left lower posterior lung field"
        assert snap["site"] == si[snap["point"]]      # current site tracks the active point
        assert "R1" not in snap["site"]               # anatomy, not the opaque code
    finally:
        c.shutdown()


def test_production_blockers_are_add_only_never_cleared():
    # Security: --production-blocker can only ADD to the fail-closed default set; no CLI
    # string (formerly the magic 'none') can clear the commissioning blockers without
    # SHA-bound activation evidence.
    server = _load_collect_server()
    default = server.DEFAULT_PRODUCTION_BLOCKERS
    assert not any("force" in blocker.lower() for blocker in default)
    assert not any("Hall" in blocker for blocker in default)
    assert server._resolve_production_blockers(None) == default
    # 'none' is no longer magic -- it is just appended as a literal, never a clear.
    assert server._resolve_production_blockers(["none"]) == default + ("none",)
    got = server._resolve_production_blockers(["extra condition"])
    assert got[: len(default)] == default          # defaults preserved, never shrunk
    assert "extra condition" in got and len(got) == len(default) + 1


def _typed_production_runtime_evidence(server):
    """Build type-correct inert evidence without opening hardware in a unit test."""

    return server.ProductionRuntimeAssembly(
        activation_path="/immutable/activation.json",
        activation_sha256="a" * 64,
        openarm_pair=object.__new__(server.OpenArmPairActivation),
        camera_exposure_contract=object.__new__(server.CameraExposureContract),
        handeye=object.__new__(server.ActivatedHandEye),
        activated_identity=object.__new__(server.ActivatedOcclusionIdentity),
        robot_capture_coordinator=object.__new__(
            server.ProductionRobotCaptureCoordinator
        ),
        robot_mask_provider=object.__new__(server.ProductionRobotMaskProvider),
        back_mask_provider=object.__new__(
            server.IndependentPosteriorBackMaskProvider
        ),
        patient_pose_provider=object.__new__(server.MetricPatientPoseProvider),
        occlusion_runtime=object.__new__(server.OcclusionRuntime),
    )


def test_complete_typed_activation_evidence_clears_only_static_blockers():
    server = _load_collect_server()
    runtime = _typed_production_runtime_evidence(server)

    assert server._resolve_production_blockers(
        None, production_runtime=runtime,
    ) == ()
    assert server._resolve_production_blockers(
        ["operator-declared blocker"], production_runtime=runtime,
    ) == ("operator-declared blocker",)


def test_missing_any_typed_activation_component_keeps_all_static_blockers():
    server = _load_collect_server()
    runtime = _typed_production_runtime_evidence(server)
    values = dict(runtime.__dict__)
    component_names = (
        "openarm_pair",
        "camera_exposure_contract",
        "handeye",
        "activated_identity",
        "robot_capture_coordinator",
        "robot_mask_provider",
        "back_mask_provider",
        "patient_pose_provider",
        "occlusion_runtime",
    )

    for missing in component_names:
        broken = server.ProductionRuntimeAssembly(
            **{**values, missing: None},
        )
        assert server._resolve_production_blockers(
            None, production_runtime=broken,
        ) == server.DEFAULT_PRODUCTION_BLOCKERS, missing

    assert server._resolve_production_blockers(
        None, production_runtime=None,
    ) == server.DEFAULT_PRODUCTION_BLOCKERS


def test_static_activation_evidence_does_not_bypass_pre_hil_robot_gate(tmp_path):
    server = _load_collect_server()
    runtime = _typed_production_runtime_evidence(server)
    static_blockers = server._resolve_production_blockers(
        None, production_runtime=runtime,
    )
    assert static_blockers == ()

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        production_blockers=static_blockers,
        start_thread=False,
    )
    try:
        assert collector._controller_ready() is False
        assert collector._device_start_rejection(
            runtime={"strict_ok": True, "blockers": []},
        ) == server._CONTROLLER_NOT_READY
    finally:
        collector.shutdown()


def test_estop_ui_button_is_not_gated_by_busy():
    # Browser buttons consume the one server-side authority.  A future accepted live
    # companion keeps stop_authorized true during bring-up; pre-HIL both remain disabled.
    server = _load_collect_server()
    page = server.PAGE
    if isinstance(page, bytes):
        page = page.decode()
    assert "async function estop(" in page                       # dedicated E-stop path
    assert "$('btnestop').onclick=()=>estop()" in page           # bound to it
    assert "ct.arm_bringup_authorized===true" in page
    assert "ct.arm_stop_authorized===true" in page
    assert "bringActive=ct.bringup_active===true" in page
    assert "$('btnarm').disabled=busy||bringActive||!bringAllowed" in page
    assert "$('btnestop').disabled=!stopAllowed" in page
    assert "hasComp=Object.prototype.hasOwnProperty.call(ct,'disabled')" in page
    # telemetry fault (arm still powered, operator decides) overrides the status line
    assert "const tf=ct.telemetry_fault;" in page
    assert "TELEMETRY FAULT" in page and "ARM STILL POWERED" in page
    assert "Safe-stop HIL required" in page


def test_bring_up_refuses_and_disables_when_fatal_races_in(tmp_path):
    # A collector-loop crash that races in DURING bring-up (after the top-of-method check)
    # must turn the bring-up into a refusal AND energy-isolate the arm, never return a
    # live-arm success onto a dead collector.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])
    disable = _ArmDisable()
    _configure_arm(c, coord, disable)
    inner_launcher = c._controller_launcher

    def racing_launcher():
        c._fatal_error = "synthetic fatal during bring-up"   # after the top check
        return inner_launcher()

    c._controller_launcher = racing_launcher
    try:
        r = c.bring_up_controller()
        assert r["ok"] is False
        assert r["code"] == "controller_bring_up_refused_fatal"
        assert disable.calls                       # arm energy-isolated, not left live
        assert c._controller_ready() is False
    finally:
        c._fatal_error = None
        c.shutdown()


def test_bring_up_refused_after_shutdown(tmp_path):
    # Once the collector is shutting down (_stop set), a fresh bring-up is refused -- never
    # energize the arm onto a stopping collector.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), _ArmDisable())
    c.shutdown()                                   # sets _stop
    r = c.bring_up_controller()
    assert r["ok"] is False
    assert r["code"] == "controller_bring_up_refused_fatal"


def test_shutdown_retries_failed_disable(tmp_path):
    # The watchdog is stopped during shutdown, so shutdown itself must retry a failed
    # disable rather than swallowing a single transient failure and leaving the arm live.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("READY", True)])
    attempts = {"n": 0}

    class _FlakyDisable:
        def __init__(self): self.calls = []
        def __call__(self, reason):
            self.calls.append(reason)
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("first disable fails")
            return type("O", (), {"outcome": "energy_isolated"})()

    _configure_arm(c, coord, _FlakyDisable())
    c.bring_up_controller()
    c.shutdown()                                   # must retry the disable itself
    assert c._controller_companion.disabled is True
    assert attempts["n"] >= 2


def _configure_safe_stop_arm(server, c, coord, disable):
    # Deliberately does NOT install the fake capability: we isolate the safe-stop path so the
    # ONLY route to real bring-up authorization is a passed operator-witnessed safe-stop HIL.
    c.require_robot = True
    c.commissioning_preview_only = False
    c._robot_coordinator = coord
    c._controller_launcher = lambda: _ArmCtrl()
    c._controller_disable = disable
    c._controller_attest = lambda pid, uid: f"attest{pid}"
    c._controller_readiness_timeout_s = 2.0
    c._controller_launch_activation = server.SimpleNamespace()   # dormant activation present
    c._safe_stop_disable_command = ("bash", "hil_disable.sh")     # candidate CAN disable (a)
    from ausc.perception.robot.safe_stop_hil import SafeStopHilGate
    c._safe_stop_gate = SafeStopHilGate(
        release_id="controller_launch_v1-" + "a" * 64,
        contract_sha256="b" * 64, session_id="test-sess",
    )


def test_real_bringup_refused_until_safe_stop_hil_passes(tmp_path):
    # Safety gate: with a production (dormant) launch activation, real bring-up is refused
    # ("Safe-stop HIL required") until the operator-witnessed safe-stop passes this session.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    _configure_safe_stop_arm(server, c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    try:
        assert c.arm_control_authorization().arm_bringup_authorized is False
        r = c.bring_up_controller()
        assert r["ok"] is False and "Safe-stop HIL" in r["message"]
        # simulate a witnessed safe-stop test (test energized, disable succeeded)
        c._safe_stop_gate.begin_test()
        c._safe_stop_gate.record_disable(command_succeeded=True, outcome="ok", controller_pid=1)
        passed = c.confirm_safe_stop(deenergized=True)
        assert passed["ok"] is True and passed["code"] == "safe_stop_hil_passed"
        assert c.arm_control_authorization().arm_bringup_authorized is True   # now authorized
    finally:
        c.shutdown()


def test_not_deenergized_witness_keeps_bringup_refused(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    _configure_safe_stop_arm(server, c, _ArmCoord([_ArmHealth("READY", True)]), _ArmDisable())
    try:
        c._safe_stop_gate.begin_test()
        c._safe_stop_gate.record_disable(command_succeeded=True, outcome="ok", controller_pid=1)
        r = c.confirm_safe_stop(deenergized=False)   # operator: arm did NOT go limp
        assert r["ok"] is False and r["code"] == "safe_stop_hil_failed"
        assert c.arm_control_authorization().arm_bringup_authorized is False
    finally:
        c.shutdown()


def test_full_safe_stop_test_flow_authorizes_real_bringup(tmp_path, monkeypatch):
    # End-to-end: operator starts the safe-stop test (energize), commands disable (recorded),
    # confirms de-energize -> passed -> real bring-up authorized. The trusted launch build is
    # stubbed to the fake arm so no real controller launches.
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    disable = _ArmDisable()
    _configure_safe_stop_arm(server, c, _ArmCoord([_ArmHealth("READY", True)]), disable)

    def fake_build(activation, *, disable_command, **_k):
        return (lambda: _ArmCtrl()), (lambda: None), disable

    monkeypatch.setattr(server, "build_launch_from_activation", fake_build)
    try:
        assert c.arm_control_authorization().arm_bringup_authorized is False
        r = c.begin_safe_stop_test()                       # 1) energize (fake)
        assert r["ok"] is True and r["safe_stop_test"] is True
        assert c._safe_stop_gate.state == "HIL_TESTING"
        d = c.disable_controller()                         # 2) disable during test
        assert d["ok"] is True
        assert c._safe_stop_gate.state == "HIL_AWAITING_WITNESS"
        p = c.confirm_safe_stop(deenergized=True)          # 3) witness de-energize
        assert p["ok"] is True and p["code"] == "safe_stop_hil_passed"
        assert c.arm_control_authorization().arm_bringup_authorized is True
        # 4) REAL second bring-up must succeed on a RENEWED receiver (the safe-stop
        # consumed the first one) and must NOT void the PASSED gate.
        renews_before = c._robot_coordinator.events.count("renew")
        r2 = c.bring_up_controller()
        assert r2["ok"] is True, r2
        assert c._robot_coordinator.events.count("renew") == renews_before + 1
        assert c._safe_stop_gate.state == "HIL_PASSED"
        assert c._controller_companion.state == "ready"
    finally:
        c.shutdown()


def test_start_robot_session_requires_arm_brought_up(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    _configure_arm(c, _ArmCoord([_ArmHealth("IDLE")]), _ArmDisable())
    try:
        with pytest.raises(RuntimeError, match="not brought up"):
            c._start_robot_session(str(tmp_path))
    finally:
        c.shutdown()


def test_start_robot_session_requires_ready_controller_even_without_callbacks(tmp_path):
    server = _load_collect_server()

    class Coordinator:
        def __init__(self):
            self.start_calls = 0

        def start_session(self, _session_dir):
            self.start_calls += 1

    coordinator = Coordinator()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=coordinator,
        start_thread=False,
    )
    try:
        with pytest.raises(RuntimeError, match="not brought up"):
            c._start_robot_session(str(tmp_path))
        assert coordinator.start_calls == 0
    finally:
        c.shutdown()


def test_state_and_start_share_exact_controller_not_ready_rejection(tmp_path):
    server = _load_collect_server()

    class Capture:
        def request_cancel(self): pass
        def start(self): pass
        def record_event(self, _payload): pass
        def abort(self, _reason): pass

    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        require_robot=True,
        robot_capture_coordinator=object(),
        start_thread=False,
    )
    c.platform.capture_factory = lambda _path: Capture()
    c.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    c._runtime_health_snapshot = lambda **_kwargs: {
        "strict_ok": True,
        "blockers": [],
        "robot": {"ready": True},
    }
    expected = "left OpenArm controller is not brought up; use Bring up arm first"
    try:
        snapshot = c.state_snapshot()
        assert snapshot["start_authorized"] is False
        assert snapshot["start_rejection"] == expected

        result = c.control("start")
        assert result["ok"] is False
        assert result["code"] == "start_not_authorized"
        assert result["message"] == expected
    finally:
        c.shutdown()


def test_state_and_post_share_back_before_audio_start_rejection(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        audio_device="synthetic-audio",
        require_audio=True,
        start_thread=False,
    )
    back = {
        "required": True,
        "confirmed": False,
        "active": False,
        "confirmable": True,
        "reason": "BACK_CONFIRMATION_REQUIRED",
    }
    c._device_start_rejection = lambda **_kwargs: None
    c._storage_start_rejection = lambda _duration: None
    c._back_confirmation_snapshot = lambda: dict(back)
    c._audio_snapshot = lambda: {
        "status": "ERROR",
        "signal_present": False,
    }
    try:
        snapshot = c.state_snapshot()
        result = c.control("start")

        assert snapshot["start_authorized"] is False
        assert snapshot["start_rejection_code"] == "back_confirmation_required"
        assert snapshot["start_rejection"] == (
            "Face away, stay alone in frame, then Confirm Back"
        )
        assert result["ok"] is False
        assert result["code"] == snapshot["start_rejection_code"]
        assert result["message"] == snapshot["start_rejection"]
    finally:
        c.shutdown()


def test_state_and_post_share_insufficient_storage_rejection(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    expected = "insufficient storage for the pinned session duration"
    c._device_start_rejection = lambda **_kwargs: None
    c._storage_start_rejection = lambda _duration: expected
    try:
        snapshot = c.state_snapshot()
        result = c.control("start")

        assert snapshot["start_authorized"] is False
        assert snapshot["start_rejection_code"] == (
            "insufficient_storage_for_planned_session"
        )
        assert snapshot["start_rejection"] == expected
        assert result["ok"] is False
        assert result["code"] == snapshot["start_rejection_code"]
        assert result["message"] == snapshot["start_rejection"]
    finally:
        c.shutdown()


def test_tail_start_storage_oserror_is_a_structured_rejection(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    calls = {"storage": 0, "capture": 0}

    def changing_storage(_duration):
        calls["storage"] += 1
        if calls["storage"] == 1:
            return None
        raise OSError("storage disappeared")

    def forbidden_capture(_path):
        calls["capture"] += 1
        raise AssertionError("capture must not be allocated after storage failure")

    c._device_start_rejection = lambda **_kwargs: None
    c._storage_start_rejection = changing_storage
    c.platform.capture_factory = forbidden_capture
    try:
        initial_state = c.platform.state.value
        result = c.control("start")
        assert result["ok"] is False
        assert result["code"] == "storage_preflight_failed"
        assert "OSError: storage disappeared" in result["message"]
        assert result["state"] == initial_state
        assert calls == {"storage": 2, "capture": 0}
    finally:
        c.shutdown()


def test_live_companion_keeps_stop_access_if_fake_capability_is_removed(tmp_path):
    server = _load_collect_server()
    c = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True,
        out_dir=tmp_path,
        start_thread=False,
    )
    disable = _ArmDisable()
    _configure_arm(c, _ArmCoord([_ArmHealth("READY", True)]), disable)
    try:
        assert c.bring_up_controller()["ok"] is True
        c._arm_control_capability = None
        authority = c.arm_control_authorization()
        assert authority.arm_bringup_authorized is False
        assert authority.arm_stop_authorized is True
        assert c.disable_controller()["ok"] is True
        assert disable.calls == ["operator_estop"]
        status = c._controller_status_payload()
        assert status["arm_bringup_authorized"] is False
        assert status["arm_stop_authorized"] is False
        assert status["disabled"] is True
        assert status["disable_command_succeeded"] is True
        assert status["energy_isolated"] is True
        assert status["disable_summary"] == "energy_isolated"
    finally:
        c.shutdown()


def test_arm_actions_inert_when_not_configured(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    try:
        assert c._controller_configured() is False        # fake -> no arm UI
        r = c.bring_up_controller()
        assert r["ok"] is False and r["code"] == "controller_not_configured"
        d = c.disable_controller()
        assert d["ok"] is False and d["code"] == "arm_stop_not_authorized"
        assert d["message"] == "Safe-stop HIL required"
        assert c._controller_status_payload()["configured"] is False
    finally:
        c.shutdown()


def test_storage_gate_estimate_and_rejection(tmp_path, monkeypatch):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector.camera_width, collector.camera_height, collector.depth_fps = 1920, 1080, 30
        collector.capture_rate_margin = 1.3
        collector.depth_compression_ratio = 8.0
        collector.non_depth_capture_bytes_per_s = 12_000_000
        collector.diagnostic_min_free_bytes = 0  # isolate the duration math
        rate = collector._estimated_capture_bytes_per_s()
        # compressed depth + non-depth allowance, times the safety margin
        assert rate == (1920 * 1080 * 2 * 30 / 8.0 + 12_000_000) * 1.3
        need = int(60 * rate)

        # Only ~59 s fits -> refuse a 60 s planned session.
        monkeypatch.setattr(server.shutil, "disk_usage",
                            lambda _p: server.SimpleNamespace(total=1 << 50, used=0, free=need - 1))
        rej = collector._storage_start_rejection(60.0)
        assert rej is not None and "insufficient storage" in rej

        # Plenty of room -> allowed.
        monkeypatch.setattr(server.shutil, "disk_usage",
                            lambda _p: server.SimpleNamespace(total=1 << 50, used=0, free=need * 4))
        assert collector._storage_start_rejection(60.0) is None

        # Gate disabled for a non-positive planned duration.
        assert collector._storage_start_rejection(0) is None
        # Unknown camera mode -> rate 0 -> gate not applicable.
        collector.camera_width = None
        assert collector._estimated_capture_bytes_per_s() == 0.0
        assert collector._storage_start_rejection(60.0) is None
    finally:
        collector.shutdown()


def test_start_refused_when_storage_cannot_hold_planned_session(tmp_path, monkeypatch):
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )

    class Capture:
        def request_cancel(self): pass
        def start(self): pass
        def record_event(self, _p): pass
        def write_frame(self, _f, _t): pass
        def finalize(self): return {"status": "diagnostic_complete", "training_eligible": False}
        def abort(self, _r): pass

    collector.platform.capture_factory = lambda _p: Capture()
    collector.platform._writer_types = {"DiagnosticPerceptionCapture": Capture}
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = lambda **_k: {"strict_ok": True, "blockers": []}
    collector._device_start_rejection = lambda **_k: None
    collector.camera_width, collector.camera_height, collector.depth_fps = 1920, 1080, 30
    collector.diagnostic_min_free_bytes = 0
    collector.planned_session_duration_s = 60.0
    rate = collector._estimated_capture_bytes_per_s()
    # Only ~30 s of headroom -> Start must refuse before opening the interval.
    monkeypatch.setattr(server.shutil, "disk_usage",
                        lambda _p: server.SimpleNamespace(total=1 << 50, used=0, free=int(30 * rate)))
    try:
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "insufficient_storage_for_planned_session"
        assert collector.platform.state.value != "RECORDING"
    finally:
        collector.shutdown()


def test_arm_interval_if_healthy_is_atomic_on_the_health_epoch(tmp_path):
    # Unit-level proof of the atomic arm: it opens ingress only when the health
    # epoch is exactly the one captured before setup, and never otherwise.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._capture_ingress_paused = True
        collector._health_generation = 7
        collector._last_runtime_strict_ok = True
        collector._device_start_rejection = lambda **_k: None

        # Healthy + matching epoch -> opens ingress.
        collector._runtime_health_snapshot = lambda **_k: {"strict_ok": True, "blockers": []}
        assert collector.arm_interval_if_healthy(7) is True
        assert collector._capture_ingress_paused is False

        # Reset; a stale expected epoch -> refuse, ingress stays paused.
        collector._capture_ingress_paused = True
        assert collector.arm_interval_if_healthy(6) is False
        assert collector._capture_ingress_paused is True

        # A degraded snapshot advances the epoch under the lock -> refuse.
        collector._health_generation = 7
        collector._last_runtime_strict_ok = True
        collector._runtime_health_snapshot = lambda **_k: {"strict_ok": False, "blockers": ["x"]}
        assert collector.arm_interval_if_healthy(7) is False
        assert collector._capture_ingress_paused is True
        assert collector._health_generation == 8  # epoch advanced on the degrade edge
    finally:
        collector.shutdown()


def test_safe_stop_witness_persists_across_restart_and_binds_to_sealed_identity(tmp_path, monkeypatch):
    # The witness validates the sealed controller + stop tool (content-addressed);
    # a restart changes neither, so the pass persists. It re-arms ONLY when the
    # sealed identity changes, and a failed witness revokes the record.
    server = _load_collect_server()
    tool = tmp_path / "stop.sh"
    tool.write_text("#!/bin/bash\ntrue\n")

    def build(out_dir):
        c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                             fake=True, out_dir=out_dir, start_thread=False)
        disable = _ArmDisable()
        _configure_safe_stop_arm(server, c, _ArmCoord([_ArmHealth("READY", True)]), disable)
        c._safe_stop_disable_command = ("bash", str(tool))
        return c, disable

    c1, disable1 = build(tmp_path)
    try:
        monkeypatch.setattr(
            server, "build_launch_from_activation",
            lambda activation, *, disable_command, **_k: ((lambda: _ArmCtrl()), (lambda: None), disable1),
        )
        assert c1.begin_safe_stop_test()["ok"] is True
        assert c1.disable_controller()["ok"] is True
        assert c1.confirm_safe_stop(deenergized=True)["ok"] is True
        assert (tmp_path / "safe_stop_witness.json").exists()
    finally:
        c1.shutdown()

    # "restart": a fresh collector restores PASSED for the same sealed identity
    # AND wires the launch/disable callables exactly like the live safe-stop test
    # (2026-08-07: a restored witness with unwired callables sent bring-up into
    # `None(...)` and the disable fallback failed the same way).
    c2, _d2 = build(tmp_path)
    try:
        c2._try_restore_safe_stop_witness()
        assert c2._safe_stop_gate.state == "HIL_PASSED"
        assert c2.arm_control_authorization().arm_bringup_authorized is True
        assert callable(c2._controller_launcher)
        assert callable(c2._controller_disable)
    finally:
        c2.shutdown()

    # wiring failure -> witness NOT restored (no authorization without a
    # working stop path), even though the persisted identity matches
    c2b, _d2b = build(tmp_path)
    try:
        monkeypatch.setattr(
            server, "build_launch_from_activation",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("launch build broken")),
        )
        c2b._controller_launcher = None
        c2b._controller_disable = None
        c2b._try_restore_safe_stop_witness()
        assert c2b._safe_stop_gate.state == "HIL_PENDING"
        assert c2b._controller_disable is None
    finally:
        c2b.shutdown()
    monkeypatch.setattr(
        server, "build_launch_from_activation",
        lambda activation, *, disable_command, **_k: ((lambda: _ArmCtrl()), (lambda: None), disable1),
    )

    # changed stop tool content -> identity mismatch -> witness NOT restored
    tool.write_text("#!/bin/bash\nfalse\n")
    c3, _d3 = build(tmp_path)
    try:
        c3._try_restore_safe_stop_witness()
        assert c3._safe_stop_gate.state == "HIL_PENDING"
    finally:
        c3.shutdown()


def test_prewarm_exercises_full_path_with_correct_contracts(tmp_path):
    # The prewarm must call detect(rgb, depth, intr) with production-resolution
    # ndarrays and assess() with the CameraStream frame contract (.frame/.index)
    # — a bare ndarray broke live on 2026-08-07 ('numpy.ndarray' has no 'index')
    # and silently skipped the warm. Both passes must complete.
    import numpy as np

    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    calls = {"detect": 0, "assess": 0}

    class _Det:
        def detect(self, rgb, depth, intr):
            assert isinstance(rgb, np.ndarray) and rgb.shape == (1080, 1920, 3)
            assert isinstance(depth, np.ndarray) and depth.dtype == np.uint16
            assert (depth == 0).any()          # holes present -> fill path warmed
            assert float(intr.fx) > 0
            calls["detect"] += 1

    class _Gate:
        def assess(self, frame):
            rgb = frame.frame                   # the REAL gate's contract
            index = int(frame.index)
            assert isinstance(rgb, np.ndarray) and index == 0
            calls["assess"] += 1
            return {}

    try:
        c._heavy_model_cache = (_Det(), _Gate())
        c._prewarm_heavy_models()
        assert calls == {"detect": 2, "assess": 2}   # both passes, both models
    finally:
        c.shutdown()


def _strided_boundary_fixture():
    """Divisor-2 recording: kept indices 10, 12, 14; End snapshot at 15
    (a full-rate camera frame one index past the last kept frame)."""

    return {
        "camera_start_boundary_index": 9,
        "camera_end_boundary_proof": {
            "boundary_latched_atomically": True,
            "post_boundary_overwrite_prevented": True,
            "snapshot_frame_index": 15,
            "snapshot_age_s": 0.03,
            "max_snapshot_age_s": 0.1,
            "thread_alive_at_boundary": True,
            "thread_error_at_boundary": None,
        },
        "camera_delivery_fifo": {
            "status": "OK",
            "thread_alive": True,
            "frame_fresh": True,
            "interval_active": False,
            "accepting": False,
            "queue_depth": 0,
            "interval_overflow_frames": 0,
            "interval_enqueued_frames": 3,
            "interval_delivered_frames": 3,
            "first_enqueued_frame_index": 10,
            "last_enqueued_frame_index": 14,
            "last_delivered_frame_index": 14,
        },
        "capture_ingress_fifo": {
            "status": "OK",
            "thread_alive": True,
            "interval_active": False,
            "queue_depth": 0,
            "overflow_rejections": 0,
            "abandoned_after_error": 0,
            "accepted": 3,
            "processed": 3,
            "first_accepted_index": 10,
            "last_accepted_index": 14,
            "last_completed_index": 14,
        },
        "last_submitted_camera_index": 14,
        "last_completed_camera_index": 14,
    }


def test_end_boundary_closure_divisor_two_strided_indices():
    server = _load_collect_server()

    proof = server.validate_capture_boundary_closure(
        _strided_boundary_fixture(),
        n_rgb_frames=3,
        expected_fps=30.0,
        final_frame_range={"first_frame_index": 10, "last_frame_index": 14},
        recording_frame_divisor=2,
    )
    assert proof["closed"] is True
    assert proof["first_frame_index"] == 10
    assert proof["last_frame_index"] == 14

    # Snapshot may trail up to divisor-1 past the last kept frame — but no more.
    over = _strided_boundary_fixture()
    over["camera_end_boundary_proof"]["snapshot_frame_index"] = 16
    with pytest.raises(RuntimeError, match="closure"):
        server.validate_capture_boundary_closure(
            over,
            n_rgb_frames=3,
            expected_fps=30.0,
            recording_frame_divisor=2,
        )

    # A strided fixture must FAIL at divisor 1 (proves the check still bites).
    with pytest.raises(RuntimeError):
        server.validate_capture_boundary_closure(
            _strided_boundary_fixture(),
            n_rgb_frames=3,
            expected_fps=30.0,
        )


def test_abort_with_boundary_closed_closes_the_ingress_interval(tmp_path):
    """record42 incident: a frame-evidence abort (boundary_closed=True) left
    the ingress interval active, wedging every later Start until restart."""

    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        assert collector._ingress_worker.begin_interval() is True
        assert collector._ingress_worker.health()["interval_active"] is True

        collector._abort_recording("test abort", boundary_closed=True)

        assert collector._ingress_worker.health()["interval_active"] is False
        # A new recording interval must open cleanly afterwards.
        assert collector._ingress_worker.begin_interval() is True
        collector._ingress_worker.close_interval()
    finally:
        collector.shutdown()


def test_update_provenance_preserves_force_and_audio_sections(tmp_path):
    """2026-08-13 incident: the camera/model/runtime rebuild dropped the
    startup force/audio sections, so the first advisory force failure at End
    KeyError'd and stranded a 14-minute session in RECORDING."""

    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    try:
        collector._provenance["force"] = {"advisory": True, "available": True}
        collector._provenance["audio"] = {"single_owner": True}
        collector._update_provenance()
        assert collector._provenance["force"] == {
            "advisory": True, "available": True,
        }
        assert collector._provenance["audio"] == {"single_owner": True}
        assert "camera" in collector._provenance

        # And the preserve path survives a provenance missing "force".
        collector._provenance.pop("force", None)
        event = collector._preserve_advisory_force_failure(
            "end_or_bind", RuntimeError("hall serial flap"),
        )
        assert event["reason"].endswith("hall serial flap")
        assert collector._provenance["force"]["last_capture_status"] == "DEGRADED"
    finally:
        collector.shutdown()


def test_displayable_points_show_partial_verified_and_amber_unverified():
    """2026-08-29 operator-display policy: the task gate stays all-or-nothing,
    but the HUMAN display shows the verified subset plus explicitly
    unverified detector-only points (robot-band-withheld L5/R5) — hiding
    eight verified landmarks from a doctor because two are uncertain starved
    the alignment workflow (live UI showed 0 points while the tracker held 8)."""
    server = _load_collect_server()
    names = [f"P{i}" for i in range(10)]
    tracking = {
        "back_present": True,
        "initialized": True,
        "reason": "TRACKING",
        "points": [
            {
                "name": name,
                "pixel": [100.0 + i, 200.0],
                "status": "TRACKING",
                "measured": True,
                "position_std": 0.001,
            }
            for i, name in enumerate(names[:8])
        ],
        "display_only_points": [
            {"name": names[8], "pixel": [500.0, 600.0], "display_only": True},
            {"name": names[9], "pixel": [520.0, 640.0], "display_only": True},
            # junk rows must be ignored
            {"name": "P0", "pixel": [1.0, 1.0], "display_only": True},
            {"name": "bad", "pixel": [float("nan"), 1.0], "display_only": True},
        ],
    }
    # task gate: still refuses a partial constellation
    assert server.visible_tracking_points(tracking, expected_names=names) == []
    verified, unverified = server.displayable_tracking_points(
        tracking, expected_names=names,
    )
    assert [p["name"] for p in verified] == names[:8]
    assert [p["name"] for p in unverified] == [names[8], names[9]]
    # duplicate-name display-only entry for a verified point was dropped
    assert all(p["name"] != "P0" for p in unverified)


def test_forensics_append_writes_flushes_and_never_raises(tmp_path):
    server = _load_collect_server()
    ok = server.Collector._append_forensics_line(
        tmp_path, "abort_forensics.jsonl", {"kind": "abort", "reason": "x"},
    )
    assert ok is True
    ok = server.Collector._append_forensics_line(
        tmp_path, "abort_forensics.jsonl",
        {"kind": "abort", "reason": "y", "weird": object()},
    )
    assert ok is True                      # default=str serializes anything
    rows = [
        json.loads(line)
        for line in (tmp_path / "abort_forensics.jsonl").read_text().splitlines()
    ]
    assert [row["reason"] for row in rows] == ["x", "y"]
    assert all("utc_ns" in row and "monotonic_t_s" in row for row in rows)
    # Never raises: missing dir, None dir
    assert server.Collector._append_forensics_line(
        tmp_path / "no" / "such" / "dir", "x.jsonl", {},
    ) is False
    assert server.Collector._append_forensics_line(None, "x.jsonl", {}) is False


def test_abort_forensics_payload_survives_broken_collectors(tmp_path):
    server = _load_collect_server()

    class _Broken:
        platform = server.SimpleNamespace(
            state=server.CollectionState.RECORDING, session_dir=str(tmp_path),
        )
        _ingress_worker = None
        _robot_coordinator = None

        def _runtime_health_snapshot(self, **_kwargs):
            raise RuntimeError("health exploded")

        def _camera_delivery_health(self):
            return {"status": "OK"}

        def _receiver_health_brief(self):
            return None

    payload = server.Collector._abort_forensics_payload(
        _Broken(), "some reason", kind="abort", extra={"boundary_closed": True},
    )
    assert payload["kind"] == "abort"
    assert payload["reason"] == "some reason"
    assert payload["boundary_closed"] is True
    assert payload["camera_delivery"] == {"status": "OK"}
    assert payload["runtime_health"].startswith("UNAVAILABLE: RuntimeError")
    assert json.dumps(payload, default=str)   # always JSON-serializable


# --- Forensic sidecar edge coverage (train #69 abort/spike persistence) ------


def _stubbed_flow_collector(server, tmp_path, capture_cls=None):
    """Fake Collector wired for a successful control('start') (pattern at
    test_end_persists_capture_model_boundaries_before_finalization)."""

    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    if capture_cls is not None:
        collector.platform.capture_factory = lambda _path: capture_cls()
        collector.platform._writer_types = {
            "DiagnosticPerceptionCapture": capture_cls,
        }
        collector.platform.completion_validator = None
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    collector._camera_status = "OK"
    collector._model_status = "RUNNING"
    collector._runtime_health_snapshot = (
        lambda **_k: {"strict_ok": True, "blockers": []})
    return collector


class _SidecarFlowCapture:
    """Minimal writer stub whose finalize seals a representative artifacts
    dict (the two forensics sidecars must never collide with artifact names)."""

    seal_artifacts = {
        "rgb.mp4": {},
        "depth.zlib": {},
        "manifest.json": {},
        "DIAGNOSTIC_FINALIZATION.json": {},
    }

    def request_cancel(self):
        pass

    def start(self):
        pass

    def record_event(self, _payload):
        pass

    def write_frame(self, _frame, _tracking):
        pass

    def set_recording_boundary(self, _payload):
        pass

    def set_finalization_context(self, _payload):
        pass

    def finalize(self):
        return {
            "status": "diagnostic_complete",
            "training_eligible": False,
            "artifacts": dict(self.seal_artifacts),
        }

    def abort(self, _reason):
        pass


def _attach_recorded_evidence_stubs(server, collector):
    """Minimal same-frame robot evidence chain so _build_recorded_robot_evidence
    runs on a fake Collector (shape from
    test_robot_upper_bracket_backpressures_the_same_camera_frame...)."""

    class Robot:
        @staticmethod
        def build_frame_evidence_inputs(
            *, frame_index, camera_host_t_s, camera_device_t_ms,
        ):
            exposure = server.SimpleNamespace(
                frame_index=frame_index,
                camera_host_t_s=camera_host_t_s,
                camera_device_t_ms=camera_device_t_ms,
                camera_id="CL8326300A0",
                camera_profile_sha256="a" * 64,
                alignment_sha256="b" * 64,
                camera_latency_sha256="c" * 64,
            )
            return server.SimpleNamespace(
                camera_exposure=exposure,
                robot_trajectory=object(),
                persisted_index={
                    "exact_action_tick_first": frame_index * 10,
                    "exact_action_tick_last": frame_index * 10 + 2,
                },
            )

    class MaskProvider:
        @staticmethod
        def build(*, exposure, trajectory, aligned_depth=None, depth_scale=1.0):
            del trajectory, aligned_depth, depth_scale
            return server.SimpleNamespace(frame_index=exposure.frame_index)

    def build_bundle(**kwargs):
        frame = kwargs["camera_frame"]
        return server.SimpleNamespace(
            index=frame.index,
            camera_frame=frame,
            robot_mask_packet=kwargs["robot_mask_packet"],
        )

    collector._robot_coordinator = Robot()
    collector._robot_mask_provider = MaskProvider()
    collector._frame_evidence_bundle_factory = build_bundle


def _evidence_frame(server, index, t):
    return server.SimpleNamespace(
        index=index, t=t, device_t_ms=1000.0 + index * 33.0,
        depth=None, depth_scale=1.0,
    )


def _scripted_monotonic(values):
    """Deterministic clock: pops scripted values, then keeps ticking so late
    callers (the forensics writer's own stamp) stay monotonic."""

    remaining = list(values)
    last = [float(values[0])]

    def fake_monotonic():
        if remaining:
            last[0] = remaining.pop(0)
        else:
            last[0] += 0.001
        return last[0]

    return fake_monotonic


def test_abort_forensics_payload_survives_broken_exception_str(tmp_path):
    # Train #70: the never-raise contract holds even when a getter's
    # exception has a hostile __str__ — the label degrades to the safe
    # repr/placeholder instead of re-raising out of the builder.
    server = _load_collect_server()

    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("boom-str")

    class _Broken:
        platform = server.SimpleNamespace(
            state=server.CollectionState.RECORDING, session_dir=str(tmp_path),
        )
        _ingress_worker = None

        def _runtime_health_snapshot(self, **_kwargs):
            raise BadStr()

        def _camera_delivery_health(self):
            return {"status": "OK"}

        def _receiver_health_brief(self):
            return None

    payload = server.Collector._abort_forensics_payload(
        _Broken(), "r", kind="abort",
    )
    assert payload["kind"] == "abort"
    assert payload["reason"] == "r"
    assert payload["runtime_health"].startswith("UNAVAILABLE: BadStr: ")
    # The unprintable message degrades to object.__repr__, never a raise.
    assert "BadStr object at" in payload["runtime_health"]
    assert payload["camera_delivery"] == {"status": "OK"}
    assert json.dumps(payload, default=str)


def test_abort_forensics_payload_collects_all_five_unavailable_labels():
    # Forensic-of-last-resort: EVERY getter fails during a wedged teardown,
    # including the untested not-None ingress-worker branch.
    server = _load_collect_server()

    class _TornIngress:
        @staticmethod
        def health():
            raise OSError("fifo torn down")

    class _Wrecked:
        # no platform attribute at all -> state getter raises AttributeError
        _ingress_worker = _TornIngress()

        def _runtime_health_snapshot(self, **_kwargs):
            raise RuntimeError("health exploded")

        def _camera_delivery_health(self):
            raise ValueError("camera gone")

        def _receiver_health_brief(self):
            raise KeyError("robot")

    payload = server.Collector._abort_forensics_payload(
        _Wrecked(), "wedged teardown", kind="abort",
    )
    assert payload["kind"] == "abort"
    assert payload["reason"] == "wedged teardown"
    assert payload["state"].startswith("UNAVAILABLE: AttributeError")
    assert payload["runtime_health"] == "UNAVAILABLE: RuntimeError: health exploded"
    assert payload["camera_delivery"] == "UNAVAILABLE: ValueError: camera gone"
    assert payload["ingress"] == "UNAVAILABLE: OSError: fifo torn down"
    # A raising KeyError formats with its repr-quoted message.
    assert payload["robot_receiver"] == "UNAVAILABLE: KeyError: 'robot'"
    assert json.dumps(payload, default=str)


def test_abort_forensics_payload_extra_semantics_as_written(tmp_path):
    server = _load_collect_server()

    class _Host:
        platform = server.SimpleNamespace(
            state=server.CollectionState.RECORDING, session_dir=str(tmp_path),
        )
        _ingress_worker = None

        def _runtime_health_snapshot(self, **_kwargs):
            return {"strict_ok": True}

        def _camera_delivery_health(self):
            return {"status": "OK"}

        def _receiver_health_brief(self):
            return None

    build = server.Collector._abort_forensics_payload
    # Train #70: extra merges FIRST, so the forensic core (kind/reason and
    # the five collected health labels) always wins over colliding keys.
    payload = build(
        _Host(), "real reason", kind="abort",
        extra={"kind": "spoofed", "reason": "spoofed", "state": "spoofed"},
    )
    assert payload["kind"] == "abort"
    assert payload["reason"] == "real reason"
    assert payload["state"] == server.CollectionState.RECORDING.value
    # Non-colliding extra keys still ride along.
    payload = build(
        _Host(), "r", kind="abort", extra={"boundary_closed": True},
    )
    assert payload["boundary_closed"] is True
    # extra=None and extra={} contribute nothing: 7 standard keys only.
    for extra in (None, {}):
        payload = build(_Host(), "r", kind="abort", extra=extra)
        assert set(payload) == {
            "kind", "reason", "state", "runtime_health", "camera_delivery",
            "ingress", "robot_receiver",
        }
    # Train #70: a non-dict extra no longer raises — it degrades to a
    # breadcrumb instead of breaking abort/start-failure handling.
    payload = build(_Host(), "r", kind="abort", extra=5)
    assert payload["invalid_extra"] == "5"
    assert payload["kind"] == "abort"


def test_non_serializable_getter_value_flows_through_writer_as_default_str(tmp_path):
    server = _load_collect_server()

    class _Host:
        platform = server.SimpleNamespace(
            state=server.CollectionState.RECORDING, session_dir=str(tmp_path),
        )
        _ingress_worker = None

        def _runtime_health_snapshot(self, **_kwargs):
            return {"strict_ok": True}

        def _camera_delivery_health(self):
            return object()

        def _receiver_health_brief(self):
            return None

    payload = server.Collector._abort_forensics_payload(
        _Host(), "r", kind="abort",
    )
    # No type validation at collection time: the object is stored verbatim.
    assert type(payload["camera_delivery"]) is object
    assert server.Collector._append_forensics_line(
        tmp_path, "abort_forensics.jsonl", payload,
    ) is True
    row = json.loads(
        (tmp_path / "abort_forensics.jsonl").read_text().splitlines()[0]
    )
    assert row["camera_delivery"].startswith("<object object at")


def test_forensics_serialization_failure_never_corrupts_the_jsonl(tmp_path):
    # default=str does NOT apply to dict keys, so a tuple-keyed payload makes
    # json.dumps raise AFTER open(); the dump runs before handle.write, so no
    # partial/torn line may reach the file.
    server = _load_collect_server()
    append = server.Collector._append_forensics_line
    assert append(
        tmp_path, "abort_forensics.jsonl", {"kind": "abort", "reason": "first"},
    ) is True
    assert append(
        tmp_path, "abort_forensics.jsonl", {("tuple", "key"): "v"},
    ) is False
    lines = (tmp_path / "abort_forensics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["reason"] == "first"
    assert append(
        tmp_path, "abort_forensics.jsonl", {"kind": "abort", "reason": "second"},
    ) is True
    lines = (tmp_path / "abort_forensics.jsonl").read_text().splitlines()
    assert [json.loads(line)["reason"] for line in lines] == ["first", "second"]


def test_forensics_non_dict_payload_fails_before_creating_the_file(tmp_path):
    # record.update(payload) runs BEFORE open(): pre-open failures create
    # nothing, while a post-open serialization failure leaves a 0-byte file.
    server = _load_collect_server()
    append = server.Collector._append_forensics_line
    assert append(tmp_path, "never.jsonl", None) is False
    assert append(tmp_path, "never.jsonl", "ab") is False
    assert not (tmp_path / "never.jsonl").exists()
    assert append(tmp_path, "torn.jsonl", {("tuple", "key"): "v"}) is False
    assert (tmp_path / "torn.jsonl").exists()
    assert (tmp_path / "torn.jsonl").read_bytes() == b""


def test_forensics_writer_timestamps_cannot_be_shadowed(tmp_path):
    # Train #70: the write-time stamps are applied AFTER the payload merge,
    # so a payload carrying utc_ns/monotonic_t_s can never masquerade as the
    # write time.
    server = _load_collect_server()
    assert server.Collector._append_forensics_line(
        tmp_path, "x.jsonl",
        {"utc_ns": "SPOOF", "monotonic_t_s": -1.0, "kind": "k"},
    ) is True
    row = json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0])
    assert isinstance(row["utc_ns"], int) and row["utc_ns"] > 0
    assert isinstance(row["monotonic_t_s"], float)
    assert row["monotonic_t_s"] >= 0.0
    assert row["kind"] == "k"


def test_forensics_numpy_scalars_survive_with_documented_coercions(tmp_path):
    # Spike-payload-shaped values: np.float64 IS a float subclass (JSON
    # number), np.int64/np.bool_ are NOT int/bool subclasses (default=str
    # stringifies them).
    server = _load_collect_server()
    assert server.Collector._append_forensics_line(
        tmp_path, "s.jsonl",
        {
            "kind": "spike",
            "frame_index": np.int64(7),
            "age_ms": np.float64(83.2),
            "flag": np.bool_(True),
        },
    ) is True
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])
    assert row["age_ms"] == 83.2
    assert row["frame_index"] == "7"
    assert row["flag"] == "True"


def test_forensics_session_dir_edge_forms(tmp_path, monkeypatch):
    server = _load_collect_server()
    append = server.Collector._append_forensics_line
    # str session_dir works (collection_platform serializes session_dir as str).
    assert append(str(tmp_path), "x.jsonl", {"kind": "k"}) is True
    row = json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0])
    assert row["kind"] == "k"
    # Empty-string session_dir: the guard fires and — critically — no stray
    # cwd-relative file appears (Path('')/'name' would resolve into the cwd).
    monkeypatch.chdir(tmp_path)
    before = sorted(entry.name for entry in tmp_path.iterdir())
    assert append("", "stray.jsonl", {"kind": "k"}) is False
    assert sorted(entry.name for entry in tmp_path.iterdir()) == before
    # session_dir pointing at an existing regular FILE: caught, unchanged.
    notadir = tmp_path / "notadir"
    notadir.write_text("keep me\n")
    assert append(notadir, "x.jsonl", {"kind": "k"}) is False
    assert notadir.read_text() == "keep me\n"


def test_forensics_append_returns_false_on_unwritable_session_dir(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("permission bits do not bind for root")
    server = _load_collect_server()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    os.chmod(sealed, 0o500)
    try:
        assert server.Collector._append_forensics_line(
            sealed, "abort_forensics.jsonl", {"kind": "abort"},
        ) is False
    finally:
        os.chmod(sealed, 0o700)
    assert list(sealed.iterdir()) == []


def test_abort_forensics_persist_before_abort_machinery_can_fail(tmp_path):
    # The entry write at 4574 runs (and fsyncs) BEFORE close_interval /
    # _abort_camera_delivery, so a wedged teardown cannot eat the evidence.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )

    def _raise_delivery():
        raise RuntimeError("delivery teardown wedged")

    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        collector._abort_camera_delivery = _raise_delivery
        with pytest.raises(RuntimeError, match="delivery teardown wedged"):
            collector._abort_recording("live fault", boundary_closed=True)
        rows = [
            json.loads(line)
            for line in (
                session_dir / "abort_forensics.jsonl"
            ).read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["kind"] == "abort"
        assert rows[0]["reason"] == "live fault"
        assert rows[0]["boundary_closed"] is True
        assert "utc_ns" in rows[0] and "monotonic_t_s" in rows[0]
    finally:
        collector.__dict__.pop("_abort_camera_delivery", None)
        collector.platform.abort("test cleanup")
        collector.shutdown()


def test_fail_start_without_session_dir_never_raises_or_writes_forensics(tmp_path):
    # Start failures before/without a session directory must silently no-op
    # the forensics write; the full cleanup still runs raise-free.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(server, tmp_path)
    real_platform = collector.platform
    try:
        assert collector.platform.session_dir is None
        result = collector._fail_start_after_platform(
            "audio dead", code="audio_dead",
        )
        assert result == {
            "ok": False, "code": "audio_dead",
            "message": "audio dead", "state": "READY",
        }
        assert list(tmp_path.rglob("abort_forensics.jsonl")) == []
        assert collector._capture_ingress_paused is True
        assert collector._audio_recording_active is False
        assert collector._force_recording_active is False

        # Platform stub lacking a session_dir attribute entirely (the
        # getattr default path); abort must be invoked and never raise.
        aborts = []
        collector.platform = server.SimpleNamespace(
            state=server.SimpleNamespace(value="READY"),
            abort=lambda reason: (
                aborts.append(reason)
                or {"ok": False, "code": "invalid_state"}
            ),
        )
        result = collector._fail_start_after_platform(
            "audio dead", code="audio_dead",
        )
        assert result["ok"] is False
        assert result["state"] == "READY"
        assert aborts == ["audio dead"]
        assert list(tmp_path.rglob("abort_forensics.jsonl")) == []
    finally:
        collector.platform = real_platform
        collector.shutdown()


def test_failed_start_writes_start_failure_forensics_record(tmp_path):
    # Shape of test_failed_start_aborts_audio_and_clears_back_confirmation,
    # but the new observable is the start_failure record on disk.
    server = _load_collect_server()

    class Capture:
        def request_cancel(self):
            pass

        def start(self):
            pass

        def set_recording_boundary(self, _payload):
            raise RuntimeError("cannot persist boundary")

        def record_event(self, _payload):
            pass

        def abort(self, _reason):
            pass

    collector = _stubbed_flow_collector(server, tmp_path, capture_cls=Capture)
    try:
        result = collector.control("start")
        assert result["ok"] is False
        assert result["code"] == "capture_start_boundary_not_persisted"
        session_dir = Path(collector.platform.session_dir)
        rows = [
            json.loads(line)
            for line in (
                session_dir / "abort_forensics.jsonl"
            ).read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["kind"] == "start_failure"
        assert rows[0]["code"] == result["code"]
        assert isinstance(rows[0]["code"], str) and rows[0]["code"]
        assert rows[0]["reason"] == result["message"]
        for section in (
            "state", "runtime_health", "camera_delivery",
            "ingress", "robot_receiver",
        ):
            assert section in rows[0]
    finally:
        collector.shutdown()


def test_abort_forensics_only_the_abortable_call_writes_a_line(
    tmp_path,
):
    # Train #70: the forensics line is written only by the call that can
    # actually tear the capture down, with the state captured AT abort entry
    # (pre-transition). A follow-on _abort_recording against the already
    # ABORTED session is a no-op (active False, platform.abort not reached)
    # and adds NO second record — same misattribution class as a
    # post-COMPLETE preview fault.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    abort_calls = []
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        original_abort = collector.platform.abort
        collector.platform.abort = lambda reason: (
            abort_calls.append(reason) or original_abort(reason)
        )
        assert collector._abort_recording("fault one") is True
        assert collector.platform.state.value == "ABORTED"
        assert collector._abort_recording("fault two") is False
        rows = [
            json.loads(line)
            for line in (
                session_dir / "abort_forensics.jsonl"
            ).read_text().splitlines()
        ]
        assert [row["reason"] for row in rows] == ["fault one"]
        assert [row["kind"] for row in rows] == ["abort"]
        assert [row["state"] for row in rows] == ["RECORDING"]
        assert [row["boundary_closed"] for row in rows] == [False]
        # platform.abort observed exactly once across both calls.
        assert abort_calls == ["fault one"]
    finally:
        collector.shutdown()


def test_finalizing_window_fault_still_records_forensics_without_abort(
    tmp_path,
):
    # Train #70 review: FINALIZING is still the CURRENT session (not yet
    # sealed), and production finalize can run for tens of seconds — a fault
    # landing in that window must keep its forensic record (this jsonl exists
    # to explain killed sessions) while the teardown still stands down:
    # platform.abort must never cancel a committed End.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    abort_calls = []
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        original_abort = collector.platform.abort
        collector.platform.abort = lambda reason: (
            abort_calls.append(reason) or original_abort(reason)
        )
        collector.platform.state = server.CollectionState.FINALIZING
        collector.platform._end_requested = True
        assert collector._abort_recording("ingress died mid-finalize") is False
        assert abort_calls == []
        rows = [
            json.loads(line)
            for line in (
                session_dir / "abort_forensics.jsonl"
            ).read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["kind"] == "abort"
        assert rows[0]["reason"] == "ingress died mid-finalize"
        assert rows[0]["state"] == "FINALIZING"
        assert rows[0]["boundary_closed"] is False
    finally:
        collector.platform.state = server.CollectionState.RECORDING
        collector.shutdown()


def test_suppressed_end_drain_abort_still_records_forensics(tmp_path):
    # The end-drain gate suppresses the abort (return False, boundary kept),
    # but the forensics append and the RUNTIME_FAULT clear both run first.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    cleared = []
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        original_clear = collector._clear_back_confirmation
        collector._clear_back_confirmation = lambda reason: (
            cleared.append(reason) or original_clear(reason)
        )
        stub = collector._ingress_worker
        close_calls_before = stub.close_calls
        assert collector._capture_ingress_paused is False
        collector._end_boundary_closing = True
        assert collector._abort_recording("late model fault") is False
        assert collector.platform.state.value == "RECORDING"
        assert stub.close_calls == close_calls_before
        assert collector._capture_ingress_paused is False
        rows = [
            json.loads(line)
            for line in (
                session_dir / "abort_forensics.jsonl"
            ).read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["kind"] == "abort"
        assert rows[0]["reason"] == "late model fault"
        assert rows[0]["boundary_closed"] is False
        assert cleared == ["RUNTIME_FAULT"]
    finally:
        collector._end_boundary_closing = False
        collector.__dict__.pop("_clear_back_confirmation", None)
        collector.platform.abort("test cleanup")
        collector.shutdown()


def test_clean_complete_flow_writes_no_forensics_sidecars(tmp_path):
    # Forensics never fires on healthy paths: full start -> end -> COMPLETE
    # with a sub-threshold recorded-evidence call in between.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    _attach_recorded_evidence_stubs(server, collector)
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        evidence, persisted = collector._build_recorded_robot_evidence(
            _evidence_frame(server, 21, time.monotonic()),
        )
        assert evidence.index == 21
        assert persisted == {
            "exact_action_tick_first": 210, "exact_action_tick_last": 212,
        }
        assert collector.control("end")["ok"] is True
        terminal = collector.platform.wait_for_terminal(timeout=2.0)
        assert terminal["state"] == "COMPLETE", terminal.get("error")
        assert not (session_dir / "abort_forensics.jsonl").exists()
        assert not (session_dir / "capture_timing_spikes.jsonl").exists()
    finally:
        collector.shutdown()


def test_finalization_tolerates_forensics_sidecars_in_session_dir(tmp_path):
    # Additive-sidecar contract: both forensics files present during End must
    # survive finalization byte-for-byte, and their names must not collide
    # with sealed artifact names.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    abort_line = '{"kind": "abort", "reason": "seeded"}\n'
    spike_line = '{"kind": "recorded_evidence_spike", "frame_index": 7}\n'
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        (session_dir / "abort_forensics.jsonl").write_text(abort_line)
        (session_dir / "capture_timing_spikes.jsonl").write_text(spike_line)
        assert collector.control("end")["ok"] is True
        terminal = collector.platform.wait_for_terminal(timeout=2.0)
        assert terminal["state"] == "COMPLETE", terminal.get("error")
        assert (session_dir / "abort_forensics.jsonl").read_text() == abort_line
        assert (
            session_dir / "capture_timing_spikes.jsonl"
        ).read_text() == spike_line
        assert server.Collector._FORENSICS_ABORT_NAME == "abort_forensics.jsonl"
        assert server.Collector._FORENSICS_SPIKES_NAME == (
            "capture_timing_spikes.jsonl"
        )
        for name in (
            server.Collector._FORENSICS_ABORT_NAME,
            server.Collector._FORENSICS_SPIKES_NAME,
        ):
            assert name not in _SidecarFlowCapture.seal_artifacts
    finally:
        collector.shutdown()


def test_post_complete_preview_fault_leaves_sealed_session_untouched(
    tmp_path,
):
    # Train #70: session_dir still points at the previous (sealed) session
    # after COMPLETE, so a later preview fault must NOT append an abort
    # record there — the write is gated on an abortable capture (active) or
    # an explicit boundary_closed caller. The preview teardown side effects
    # (pause ingress, close the preview interval, return False without
    # platform.abort) are unchanged.
    server = _load_collect_server()
    collector = _stubbed_flow_collector(
        server, tmp_path, capture_cls=_SidecarFlowCapture,
    )
    abort_calls = []
    try:
        assert collector.control("start")["ok"] is True
        session_dir = Path(collector.platform.session_dir)
        assert collector.control("end")["ok"] is True
        terminal = collector.platform.wait_for_terminal(timeout=2.0)
        assert terminal["state"] == "COMPLETE", terminal.get("error")
        stub = collector._ingress_worker
        close_calls_before = stub.close_calls
        original_abort = collector.platform.abort
        collector.platform.abort = lambda reason: (
            abort_calls.append(reason) or original_abort(reason)
        )
        assert collector._abort_recording("preview camera stale") is False
        assert abort_calls == []
        assert collector._capture_ingress_paused is True
        assert stub.close_calls == close_calls_before + 1
        assert not (session_dir / "abort_forensics.jsonl").exists()
    finally:
        collector.shutdown()


def test_recorded_evidence_spike_threshold_is_strictly_greater_on_total(
    tmp_path, monkeypatch,
):
    # Line 5227: `_frame_age_ms > 80.0 or (_t2 - _t0) * 1000.0 > 80.0` —
    # strict > on the frame age and on the TOTAL bracket+bundle time (never
    # the individual parts).
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    _attach_recorded_evidence_stubs(server, collector)
    real_monotonic = time.monotonic
    try:
        for name, frame_t, script, expect_spike in (
            # age exactly 80.0ms AND total exactly 80.0ms: both strict
            # comparisons fail (binary-clean bases: 0.08/0.16).
            ("caseA", 0.0, [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.16], False),
            # age 80.1ms with total 62.5ms: the age term alone trips.
            ("caseB", 0.0,
             [0.0801, 0.09, 0.10, 0.11, 0.12, 0.13, 0.1426], True),
            # age 0, bracket 50ms + bundle 40ms: each printed part < 80 but
            # the TOTAL (90ms) trips.
            ("caseC", 1.0, [1.0, 1.05, 1.05, 1.06, 1.06, 1.08, 1.09], True),
        ):
            session = tmp_path / name
            session.mkdir()
            collector.platform.session_dir = session
            collector._last_recorded_evidence_report_t = 0.0
            monkeypatch.setattr(
                server.time, "monotonic", _scripted_monotonic(script),
            )
            evidence, persisted = collector._build_recorded_robot_evidence(
                _evidence_frame(server, 31, frame_t),
            )
            assert evidence.index == 31
            assert persisted == {
                "exact_action_tick_first": 310, "exact_action_tick_last": 312,
            }
            spike_path = session / "capture_timing_spikes.jsonl"
            assert spike_path.exists() is expect_spike, name
            if expect_spike:
                rows = [
                    json.loads(line)
                    for line in spike_path.read_text().splitlines()
                ]
                assert len(rows) == 1
                assert rows[0]["kind"] == "recorded_evidence_spike"
        # Case C detail: the recorded parts are each under the threshold —
        # the spike fired on their total.
        row = json.loads(
            (tmp_path / "caseC" / "capture_timing_spikes.jsonl")
            .read_text().splitlines()[0]
        )
        assert row["bracket_ms"] == 50.0
        assert row["bundle_ms"] == 40.0
    finally:
        # The scripted clock patched the shared stdlib module; put the real
        # one back before shutdown joins worker threads.
        server.time.monotonic = real_monotonic
        collector.platform.session_dir = None
        collector.shutdown()


def test_recorded_evidence_spike_record_fields_and_zero_mask_canary(
    tmp_path, monkeypatch,
):
    # Record shape: exactly the 7 payload keys plus the writer's two stamps.
    # Train #70: -1.0 is reserved for "never measured" (is None); mask AND
    # factory builds measuring exactly 0.0ms are recorded truthfully as 0.0.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    _attach_recorded_evidence_stubs(server, collector)
    session = tmp_path / "spike_session"
    session.mkdir()
    collector.platform.session_dir = session
    # mask build frozen at 0.10 AND factory frozen at 0.12 -> both 0.0 exactly.
    script = [0.0801, 0.09, 0.10, 0.10, 0.12, 0.12, 0.1426]
    real_monotonic = time.monotonic
    try:
        monkeypatch.setattr(
            server.time, "monotonic", _scripted_monotonic(script),
        )
        collector._build_recorded_robot_evidence(
            _evidence_frame(server, 31, 0.0),
        )
        rows = [
            json.loads(line)
            for line in (
                session / "capture_timing_spikes.jsonl"
            ).read_text().splitlines()
        ]
        assert len(rows) == 1
        row = rows[0]
        assert set(row) == {
            "kind", "frame_index", "age_at_bracket_ms", "bracket_ms",
            "bundle_ms", "mask_ms", "factory_ms", "utc_ns", "monotonic_t_s",
        }
        assert row["kind"] == "recorded_evidence_spike"
        assert row["frame_index"] == 31
        assert row["age_at_bracket_ms"] == 80.1
        assert row["bracket_ms"] == round((0.09 - 0.0801) * 1000.0, 1)
        assert row["bundle_ms"] == round((0.1426 - 0.09) * 1000.0, 1)
        assert row["factory_ms"] == 0.0
        assert row["mask_ms"] == 0.0
    finally:
        server.time.monotonic = real_monotonic
        collector.platform.session_dir = None
        collector.shutdown()


def test_spike_write_failure_never_breaks_the_evidence_path(tmp_path, capsys):
    # platform.session_dir pointing at a regular FILE makes the spike append
    # fail inside _append_forensics_line; the evidence hot path must still
    # return normally and the SPIKE print must still occur.
    server = _load_collect_server()
    collector = server.Collector(
        [server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
        fake=True, out_dir=tmp_path, start_thread=False,
    )
    collector._ingress_worker.stop()
    collector._ingress_worker = _IngressStub()
    _attach_recorded_evidence_stubs(server, collector)
    notadir = tmp_path / "notadir"
    notadir.write_text("keep me\n")
    collector.platform.session_dir = notadir
    try:
        evidence, persisted = collector._build_recorded_robot_evidence(
            _evidence_frame(server, 5, time.monotonic() - 0.5),
        )
        assert evidence.index == 5
        assert persisted == {
            "exact_action_tick_first": 50, "exact_action_tick_last": 52,
        }
        assert "[record-evidence-SPIKE]" in capsys.readouterr().out
        assert notadir.read_text() == "keep me\n"
        assert list(tmp_path.rglob("capture_timing_spikes.jsonl")) == []
    finally:
        collector.platform.session_dir = None
        collector.shutdown()
