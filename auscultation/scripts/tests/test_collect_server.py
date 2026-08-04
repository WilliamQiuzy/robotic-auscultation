from __future__ import annotations

import importlib.util
import http.client
import hashlib
import json
import math
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

    result = server.tracking_for_preview_frame(5, tracking)

    assert result["back_present"] is False
    assert result["constellation_valid"] is False
    assert result["reason"] == "MODEL_PENDING_FOR_CURRENT_FRAME"
    assert result["points"] == []


def test_async_model_preview_uses_its_exact_source_frame_not_newer_camera_frame(
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

        @staticmethod
        def health():
            return {"status": "OK", "error": None}

    collector.fake = False
    collector._model_worker = Worker()
    collector._frame_ring[4] = source
    try:
        assert collector._consume_model_result() is True

        selected = collector._select_preview_source(
            newer,
            collector._latest_tracking,
            model_result_updated=True,
        )

        assert selected is not None
        preview_frame, preview_tracking = selected
        assert preview_frame is source
        assert preview_tracking["frame_index"] == source.index
        assert preview_tracking["reason"] == "TRACKING"
        assert [point["name"] for point in preview_tracking["points"]] == ["R1"]
        assert collector._select_preview_source(
            newer,
            collector._latest_tracking,
            model_result_updated=False,
        ) is None
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
        "--hall-calibration", str(tmp_path / "hall-calibration.json"),
        "--hall-calibration-sha256", "1" * 64,
        "--hall-port", (
            "/dev/serial/by-id/"
            "usb-MicroPython_Board_in_FS_mode_53032847188a2e1c-if00"
        ),
        "--hall-expected-usb-vid", "0x2e8a",
        "--hall-expected-usb-pid", "0x0005",
        "--hall-expected-usb-serial", "53032847188a2e1c",
        "--sync-calibration", str(tmp_path / "sync-calibration.json"),
        "--sync-calibration-sha256", "2" * 64,
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
        def __init__(self, person_gate):
            assert isinstance(person_gate, PersonGate)

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
    assert "<strong>COMMISSIONING</strong><span>NOT TRAINING-READY</span>" in page
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
    assert "await sha256Hex(bytes)!==expectedDigest" in page
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


def test_required_force_implies_commissioned_audio_and_pinned_channel(tmp_path):
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
        assert (
            "Hall-force synchronization requires commissioned stethoscope audio"
            in blockers
        )
        assert "Hall-force synchronization requires a pinned audio channel" in blockers
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
        "Training gates", "Audio-force sync", "Camera latency", "Motor stop",
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
    # not the raw depth-mode size, so the conservative rate uses 1920x1080.
    expected_rate = 1920 * 1080 * 2 * 30
    assert storage["status"] == "OK"
    assert storage["free_bytes"] == free
    assert storage["usable_bytes_before_safety_reserve"] == 50 * 1024 ** 3
    assert storage["conservative_bytes_per_second"] == expected_rate
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
        "policy_version": "audio_signal_conservative_v2",
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

    class MaskProvider:
        @staticmethod
        def build(*, exposure, trajectory):
            del trajectory
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
        def build(*, exposure, trajectory):
            del trajectory
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
    original_reset = collector._reset_camera_pipeline

    def observed_reset(reason):
        original_reset(reason)
        reset_evidence.append({
            "reason": str(reason),
            "points": list(collector._current_tracking.get("points") or []),
            "model_worker": collector._model_worker,
            "camera_status": collector._camera_status,
            "at": time.monotonic(),
        })

    collector._reset_camera_pipeline = observed_reset
    collector.start()
    try:
        assert _wait_until(lambda: bool(reset_evidence), timeout=1.5)
        assert "simulated CUDA worker failure" in reset_evidence[0]["reason"]
        assert reset_evidence[0]["points"] == []
        assert reset_evidence[0]["model_worker"] is None
        assert reset_evidence[0]["camera_status"] == "RECONNECTING"
        assert cameras[0].stopped is True

        assert _wait_until(
            lambda: len(cameras) >= 2
            and len(processor_generations) >= 2
            and collector._camera_status == "OK"
            and collector._model_status == "RUNNING",
            timeout=2.0,
        )
        assert attempts[1] - reset_evidence[0]["at"] >= 0.06
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
        assert _wait_until(lambda: collector._camera_status == "RECONNECTING")
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
    collector.require_robot = True
    collector.commissioning_preview_only = False
    collector._robot_coordinator = coord
    collector._controller_launcher = lambda: _ArmCtrl()
    collector._controller_disable = disable
    collector._controller_attest = lambda pid, uid: f"attest{pid}"
    collector._controller_readiness_timeout_s = 2.0


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
        assert coord.events[:3] == ["prepare", "attach", "start"]
        assert disable.calls == []                       # never disabled on success
        d = c.disable_controller()
        assert d["ok"] and d["code"] == "controller_disabled"
        assert disable.calls == ["operator_estop"]
        assert c._controller_ready() is False
    finally:
        c.shutdown()


def test_disable_without_motor_readback_is_not_energy_isolated(tmp_path):
    # P0-2: with no independent motor-disable readback wired, a successful disable
    # command must surface as command-succeeded/UNVERIFIED, never "energy-isolated".
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    coord = _ArmCoord([_ArmHealth("WARMING"), _ArmHealth("READY", True)])
    _configure_arm(c, coord, _ArmDisable())
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
    try:
        r = c.bring_up_controller()
        assert r["ok"] is False and r["code"] == "controller_bring_up_faulted"
        assert len(disable.calls) == 1                   # disable commanded on failure
        assert r["energy_isolated"] is False             # P0-2: unverified w/o readback
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


def test_collector_loop_crash_disables_arm(tmp_path):
    # Safety: a dead collector loop energy-isolates the arm (disable-first) -- a real
    # software safety fault DOES stop the arm (distinct from an advisory force reading).
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
        # disable-first: even if data teardown throws, the arm is still disabled.
        c._abort_recording = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("abort boom"))
        c._loop_guarded()
        assert disable.calls                       # the arm was disabled FIRST
        assert c._controller_companion.disabled is True
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


def test_watchdog_thread_disables_arm_on_telemetry_fault(tmp_path):
    # Safety: controller supervision runs on an INDEPENDENT watchdog thread, so a
    # telemetry fault energy-isolates the arm even if the main loop is busy or hung.
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
        while not c._controller_companion.disabled and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert c._controller_companion.disabled is True  # watchdog disabled the arm
        assert disable.calls
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


def test_watchdog_retries_failed_disable_until_confirmed(tmp_path):
    # Safety: if the first automatic disable command FAILS (arm faulted but not confirmed
    # disabled), the watchdog must keep re-issuing the disable until it succeeds -- one
    # transient disable failure can never leave the arm energized.
    import threading as _threading
    import time as _time
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
    c._controller_watchdog_interval_s = 0.005
    try:
        c.bring_up_controller()
        assert c._controller_ready() is True
        wd = _threading.Thread(target=c._controller_watchdog_run, daemon=True)
        wd.start()
        coord._script = [_ArmHealth("STALE", False)]      # telemetry fault -> disable
        deadline = _time.monotonic() + 2.0
        while not c._controller_companion.disabled and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert c._controller_companion.disabled is True   # eventually confirmed disabled
        assert attempts["n"] >= 2                          # retried after the first failure
    finally:
        c._stop.set()
        wd.join(timeout=1.0)
        c.shutdown()


def test_production_blockers_are_add_only_never_cleared():
    # Security: --production-blocker can only ADD to the fail-closed default set; no CLI
    # string (formerly the magic 'none') can clear the commissioning blockers without
    # SHA-bound activation evidence.
    server = _load_collect_server()
    default = server.DEFAULT_PRODUCTION_BLOCKERS
    assert server._resolve_production_blockers(None) == default
    # 'none' is no longer magic -- it is just appended as a literal, never a clear.
    assert server._resolve_production_blockers(["none"]) == default + ("none",)
    got = server._resolve_production_blockers(["extra condition"])
    assert got[: len(default)] == default          # defaults preserved, never shrunk
    assert "extra condition" in got and len(got) == len(default) + 1


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


def test_arm_actions_inert_when_not_configured(tmp_path):
    server = _load_collect_server()
    c = server.Collector([server.SimpleNamespace(name="R1", uv=(0.2, 0.2))],
                         fake=True, out_dir=tmp_path, start_thread=False)
    try:
        assert c._controller_configured() is False        # fake -> no arm UI
        r = c.bring_up_controller()
        assert r["ok"] is False and r["code"] == "controller_not_configured"
        d = c.disable_controller()
        assert d["ok"] and d["code"] == "controller_not_running"
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
