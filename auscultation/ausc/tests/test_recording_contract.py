"""Pure contract tests for real multimodal data collection (no devices required)."""

from __future__ import annotations

import copy
import hashlib
import inspect
import sys
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

AUSC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUSC))

import record_session as rec  # noqa: E402


FORCE_SHA256 = "a" * 64
SYNC_CAPTURE_MODE = {
    "audio_channel": 1,
    "force_calibration_sha256": FORCE_SHA256,
    "audio": {
        "requested_device": "Scarlett Solo USB",
        "portaudio_index": 3,
        "hostapi_name": "ALSA",
        "device_name": "Scarlett Solo USB Audio",
        "max_input_channels": 2,
        "usb_serial": "AUDIO-SERIAL-001",
        "usb_vid": 0x1235,
        "usb_pid": 0x8211,
        "usb_identity_source": "linux_alsa_sysfs",
    },
    "hall": {
        "requested_port": "/dev/cu.usbmodem101",
        "device": "/dev/cu.usbmodem101",
        "usb_serial": "HALL-SERIAL-001",
        "usb_vid": 0x2E8A,
        "usb_pid": 0x000A,
        "manufacturer": "Raspberry Pi",
        "product": "Pico",
        "location": "1-2",
    },
}
CAMERA_CAPTURE_MODE = {
    "backend": "orbbec",
    "camera_id": "0",
    "width": 640,
    "height": 480,
    "fps": 30.0,
    "time_source": "orbbec_device_and_host_arrival",
    "color_format": "RGB",
    "depth_mode": {"width": 640, "height": 576, "fps": 30.0, "format": "Y16"},
    "align_mode": "software_depth_to_color",
    "frame_aggregate_mode": "full_frame_require",
}


def test_record_session_default_audio_query_selects_only_scarlett_alsa_input():
    parameter = inspect.signature(rec.record_session).parameters["audio_device"]
    assert parameter.default == "Scarlett Solo USB: Audio"


def test_recorder_refuses_nonempty_session_without_touching_existing_artifacts(tmp_path):
    session = tmp_path / "reserved"
    session.mkdir()
    existing = session / "targets.jsonl"
    existing.write_bytes(b"irreplaceable-existing-session\n")

    with pytest.raises(FileExistsError, match="non-empty.*refusing to overwrite"):
        rec.reserve_session_directory(tmp_path, session_dir=session)

    assert existing.read_bytes() == b"irreplaceable-existing-session\n"
    assert sorted(path.name for path in session.iterdir()) == ["targets.jsonl"]


def test_recorder_accepts_only_an_explicit_preallocated_empty_session(tmp_path):
    preallocated = tmp_path / "supervisor-reserved"
    preallocated.mkdir()

    assert rec.reserve_session_directory(tmp_path, session_dir=preallocated) == preallocated

    auto_collision = tmp_path / "session_collision"
    auto_collision.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        rec.reserve_session_directory(
            tmp_path, session_dir=None, generated_name="session_collision",
        )


def test_record_session_rejects_existing_artifact_before_hardware_or_manifest_write(tmp_path):
    session = tmp_path / "existing-session"
    session.mkdir()
    manifest = session / "manifest.json"
    original = b'{"status":"complete","training_eligible":true}\n'
    manifest.write_bytes(original)

    with pytest.raises(FileExistsError, match="non-empty.*refusing to overwrite"):
        rec.record_session(
            str(tmp_path), 1.0, session_dir=str(session),
            hall_cal=str(tmp_path / "does-not-matter.json"),
        )

    assert manifest.read_bytes() == original


def test_rgbd_storage_budget_uses_negotiated_mode_duration_and_deferred_writes():
    mode = {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "depth_mode": {"width": 640, "height": 576, "fps": 30.0, "format": "Y16"},
        "align_mode": "software_depth_to_color",
    }

    short = rec.estimate_capture_storage(
        mode, backend="orbbec", seconds=10.0, control_hz=50.0,
    )
    long = rec.estimate_capture_storage(
        mode, backend="orbbec", seconds=20.0, control_hz=50.0,
    )

    assert short["mode"] == {
        "width": 1920, "height": 1080, "fps": 30.0,
        "depth_width": 1920, "depth_height": 1080,
        "depth_bytes_per_pixel": 2,
    }
    # Depth is budgeted at the conservative compression floor, not raw: the stream
    # estimate must exceed rgb + compressed-depth yet stay below rgb + raw-depth.
    rgb_bytes = 1920 * 1080 * 1 * 30 * 10
    raw_depth_bytes = 1920 * 1080 * 2 * 30 * 10
    compressed_depth_floor = raw_depth_bytes / rec.DEPTH_CONSERVATIVE_COMPRESSION_RATIO
    assert short["streaming_bytes"] > rgb_bytes + compressed_depth_floor
    assert short["streaming_bytes"] < rgb_bytes + raw_depth_bytes
    assert short["depth_compression_ratio"] == rec.DEPTH_CONSERVATIVE_COMPRESSION_RATIO
    assert short["deferred_bytes"] >= 16_000 * 2 * 10
    assert short["estimated_capture_bytes"] == (
        short["streaming_bytes"] + short["deferred_bytes"]
    )
    assert long["estimated_capture_bytes"] > 1.9 * short["estimated_capture_bytes"]


def test_storage_guard_rejects_impossible_start_and_persists_failure(tmp_path):
    usage = namedtuple("usage", "total used free")
    budget = {
        "estimated_capture_bytes": 800,
        "streaming_bytes": 600,
        "deferred_bytes": 200,
    }
    guard = rec.StorageGuard(
        tmp_path, budget=budget, duration_s=10.0, safety_reserve_bytes=300,
        disk_usage=lambda _path: usage(10_000, 9_000, 1_000),
    )

    with pytest.raises(rec.RecordingAborted, match="storage budget.*1100.*1000"):
        guard.check_start()

    failure = json.loads((tmp_path / "STORAGE_ABORT.json").read_text())
    assert failure["phase"] == "start"
    assert failure["required_free_bytes"] == 1_100
    assert failure["available_free_bytes"] == 1_000


def test_runtime_storage_watermark_aborts_and_manifest_stays_failed(tmp_path):
    usage = namedtuple("usage", "total used free")
    free_values = iter([2_000, 700])
    guard = rec.StorageGuard(
        tmp_path,
        budget={
            "estimated_capture_bytes": 1_000,
            "streaming_bytes": 800,
            "deferred_bytes": 200,
        },
        duration_s=10.0,
        safety_reserve_bytes=400,
        disk_usage=lambda _path: usage(10_000, 8_000, next(free_values)),
        check_interval_s=0.0,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {"status": "initializing", "training_eligible": False}

    with pytest.raises(rec.RecordingAborted, match="runtime storage watermark"):
        with rec._ManifestLifecycle(manifest_path, manifest) as lifecycle:
            guard.check_start()
            lifecycle.recording(100.0)
            guard.check_runtime(elapsed_s=5.0)

    persisted = json.loads(manifest_path.read_text())
    failure = json.loads((tmp_path / "STORAGE_ABORT.json").read_text())
    assert persisted["status"] == "aborted"
    assert persisted["training_eligible"] is False
    assert "runtime storage watermark" in persisted["abort_reason"]
    assert failure["phase"] == "recording"
    assert failure["required_free_bytes"] == 1_000
    assert failure["available_free_bytes"] == 700


def test_runtime_projection_aborts_early_when_compression_worse_than_budget(tmp_path):
    usage = namedtuple("usage", "total used free")
    # start sees 2000 free (needs 1400 -> passes); at 50s of 100s only 1100 free
    # remains (900 used).  The classic watermark still passes (needs 1000), but
    # projecting 900 bytes/50s to the full 100s = 1800 would breach the 400 reserve.
    free_values = iter([2_000, 1_100])
    guard = rec.StorageGuard(
        tmp_path,
        budget={
            "estimated_capture_bytes": 1_000,
            "streaming_bytes": 800,
            "deferred_bytes": 200,
        },
        duration_s=100.0,
        safety_reserve_bytes=400,
        disk_usage=lambda _path: usage(10_000, 8_000, next(free_values)),
        check_interval_s=0.0,
        projection_warmup_s=30.0,
    )
    guard.check_start()
    with pytest.raises(rec.RecordingAborted, match="projected capture rate"):
        guard.check_runtime(elapsed_s=50.0)
    failure = json.loads((tmp_path / "STORAGE_ABORT.json").read_text())
    assert failure["phase"] == "recording_projection"


def test_runtime_projection_does_not_trip_a_normal_session(tmp_path):
    usage = namedtuple("usage", "total used free")
    # real capture runs well under budget (200 used in 50s -> projects to 400,
    # leaving 1600 free): neither the watermark nor the projection may abort.
    free_values = iter([2_000, 1_800])
    guard = rec.StorageGuard(
        tmp_path,
        budget={
            "estimated_capture_bytes": 1_000,
            "streaming_bytes": 800,
            "deferred_bytes": 200,
        },
        duration_s=100.0,
        safety_reserve_bytes=400,
        disk_usage=lambda _path: usage(10_000, 8_000, next(free_values)),
        check_interval_s=0.0,
        projection_warmup_s=30.0,
    )
    guard.check_start()
    result = guard.check_runtime(elapsed_s=50.0)
    assert result is not None
    assert not (tmp_path / "STORAGE_ABORT.json").exists()


def test_record_session_checks_actual_negotiated_mode_budget_before_ready(
    monkeypatch, tmp_path,
):
    force_mod = ModuleType("force_sensor")

    class Calibration:
        calibrated = True
        max_screened_force_n = 4.6
        hard_limit_force_n = 5.5

        @classmethod
        def from_bytes(cls, _raw):
            return cls()

    class Force:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return SimpleNamespace(status="OK", force_n=0.0, t=1.0, device_t_ms=1.0)

    force_mod.ForceCalibration = Calibration
    force_mod.ForceSensor = Force
    monkeypatch.setitem(sys.modules, "force_sensor", force_mod)

    audio_mod = ModuleType("audio_stream")

    audio_opened = []

    class Audio:
        def __init__(self, *_args, **_kwargs):
            audio_opened.append(True)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def level_db(self):
            return SimpleNamespace(status="OK", rms_db=-20.0, peak_db=-10.0)

    audio_mod.AudioStream = Audio
    monkeypatch.setitem(sys.modules, "audio_stream", audio_mod)

    camera_mod = ModuleType("camera_stream")

    class Camera(Audio):
        def __init__(self, *_args, **_kwargs):
            pass

        negotiated_mode = {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "depth_mode": {
                "width": 640, "height": 576, "fps": 30.0, "format": "Y16",
            },
            "align_mode": "software_depth_to_color",
        }

    camera_mod.CameraStream = Camera
    monkeypatch.setitem(sys.modules, "camera_stream", camera_mod)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        rec.shutil, "disk_usage", lambda _path: usage(10_000, 9_999, 1),
    )
    calibration = tmp_path / "commissioned-force.json"
    calibration.write_text("{}")
    session = tmp_path / "new-session"

    with pytest.raises(rec.RecordingAborted, match="storage budget"):
        rec.record_session(
            str(tmp_path), 600.0, session_dir=str(session),
            hall_cal=str(calibration), hall_port="/dev/fake",
            camera_backend="orbbec", storage_reserve_bytes=1_000,
        )

    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["storage_budget"]["mode"]["width"] == 1920
    assert manifest["storage_budget"]["mode"]["depth_width"] == 1920
    assert (session / "STORAGE_ABORT.json").is_file()
    assert audio_opened == [], "impossible captures must fail before O(duration) audio allocation"


def test_end_audio_snapshot_clips_to_capture_and_cannot_be_replaced_by_late_callbacks():
    source = np.arange(30, dtype=np.float32)
    calls = []

    class Audio:
        def latest_window(self, seconds):
            calls.append(seconds)
            return SimpleNamespace(
                samples=source, sr=10, t_start=9.0, t_end=12.0,
                status="NO_DATA",
            )

    snapshot = rec.snapshot_audio_at_end(Audio(), t0=10.0, t_end=12.0)
    source[:] = -1.0  # A callback mutates/reuses its source after End.

    assert calls == [pytest.approx(5.0)]
    assert snapshot.t_start == pytest.approx(10.0)
    assert snapshot.t_end == pytest.approx(12.0)
    assert snapshot.sr == 10
    assert snapshot.status == "OK"
    assert snapshot.samples.tolist() == list(np.arange(10, 30, dtype=np.float32))


def _sync_payload() -> dict:
    return {
        "schema_version": "wam_sync_v1",
        "offset_definition": "force_t_minus_audio_t",
        "audio_to_force_offset_ms": 12.5,
        "offset_std_ms": 1.2,
        "n_matched": 4,
        "unique_matches": True,
        "measurement_method": "physical_diaphragm_taps",
        "capture_mode": copy.deepcopy(SYNC_CAPTURE_MODE),
    }


def _camera_payload() -> dict:
    return {
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure",
        "latency_ms": 12.0,
        "latency_std_ms": 1.0,
        "n_samples": 20,
        "measurement_method": "hardware_led_flash",
        "capture_mode": dict(CAMERA_CAPTURE_MODE),
    }


def test_default_force_calibration_uses_linbox_persistent_path(monkeypatch, tmp_path):
    monkeypatch.delenv("WAM_HALL_CALIBRATION", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert rec.default_hall_calibration() == (
        tmp_path / "wam_data/config/calibration/force_feedback_calibration_veo6_tared.json"
    )


def test_cooperative_stop_is_session_bound_and_does_not_depend_on_signal(tmp_path):
    stop = tmp_path / "STOP_CAPTURE.json"
    assert rec.capture_should_stop(stop, t0=10.0, max_duration_s=30.0, now=12.0) is False
    stop.write_text(json.dumps({"session_id": "session-a", "reason": "operator_end"}))
    assert rec.capture_should_stop(stop, t0=10.0, max_duration_s=30.0, now=12.0) is True


def test_capture_ended_ack_records_actual_duration_atomically(tmp_path):
    ack = tmp_path / "CAPTURE_ENDED.json"
    payload = rec.publish_capture_ended(
        ack, t0=100.25, t_end=112.75, reason="operator_end",
    )
    assert payload == {
        "schema_version": "wam_capture_ended_v1",
        "reason": "operator_end",
        "t0_monotonic": 100.25,
        "t_end_monotonic": 112.75,
        "actual_duration_s": 12.5,
    }
    assert json.loads(ack.read_text()) == payload
    assert not ack.with_suffix(".json.tmp").exists()


def test_video_writer_open_failure_aborts_before_claiming_recording(tmp_path):
    class ClosedWriter:
        def isOpened(self):
            return False

    with pytest.raises(rec.RecordingAborted, match="video writer failed to open"):
        rec.open_video_writer(
            lambda *_args: ClosedWriter(), tmp_path / "rgb.mp4",
            fourcc=123, fps=30.0, size=(640, 480),
        )


def test_force_calibration_environment_override_precedes_persistent_default(
    monkeypatch, tmp_path,
):
    selected = tmp_path / "commissioned.json"
    selected.write_text("{}")
    monkeypatch.setenv("WAM_HALL_CALIBRATION", str(selected))
    assert rec.default_hall_calibration() == selected


def test_selected_force_calibration_must_exist_and_be_regular(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Hall calibration"):
        rec.resolve_hall_calibration(missing)
    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        rec.resolve_hall_calibration(directory)


def test_uncalibrated_force_requires_explicit_rehearsal_override():
    cal = SimpleNamespace(calibrated=False)
    with pytest.raises(RuntimeError, match="calibrated=false"):
        rec.validate_force_calibration(cal, allow_uncalibrated=False)
    rec.validate_force_calibration(cal, allow_uncalibrated=True)


def test_real_collection_requires_all_three_sensor_streams():
    streams = {"force": object()}
    assert rec.required_stream_failures(streams, {"force", "audio", "cam"}) == ["audio", "cam"]


@pytest.mark.parametrize(
    ("allow_uncalibrated", "allow_missing"), [(True, False), (False, True), (True, True)],
)
def test_arm_connected_collection_forbids_rehearsal_overrides(allow_uncalibrated, allow_missing):
    with pytest.raises(ValueError, match="arm-connected.*rehearsal"):
        rec.validate_collection_mode("left", allow_uncalibrated, allow_missing)


def test_arm_collection_requires_physical_backend_and_explicit_camera_identity():
    with pytest.raises(ValueError, match="synthetic camera"):
        rec.validate_collection_mode(
            "left", False, False, camera_backend="fake", camera_id="synthetic",
        )
    with pytest.raises(ValueError, match="serial-verifiable Orbbec"):
        rec.validate_collection_mode(
            "left", False, False, camera_backend="avfoundation", camera_id="camera-0",
        )
    with pytest.raises(ValueError, match="stable camera_id"):
        rec.validate_collection_mode(
            "left", False, False, camera_backend="orbbec", camera_id=None,
        )
    rec.validate_collection_mode(
        "left", False, False, camera_backend="orbbec", camera_id="femto-bolt-01",
    )


def test_sensor_recorder_cannot_mark_combined_session_complete(tmp_path):
    manifest = {"status": "initializing", "training_eligible": False}
    path = tmp_path / "manifest.json"
    with rec._ManifestLifecycle(path, manifest) as lifecycle:
        lifecycle.sensors_complete()
    payload = json.loads(path.read_text())
    assert payload["status"] == "sensors_complete"
    assert payload["training_eligible"] is False


def test_load_sync_calibration_returns_hashed_signed_contract(tmp_path):
    path = tmp_path / "sync.json"
    path.write_text(json.dumps(_sync_payload()))
    meta = rec.load_sync_calibration(path, expected_capture_mode=SYNC_CAPTURE_MODE)
    assert meta["calibration_path"] == str(path.resolve())
    assert len(meta["calibration_sha256"]) == 64
    assert meta["audio_to_force_offset_ms"] == pytest.approx(12.5)
    assert meta["offset_units"] == "ms"
    assert meta["applied"] is False
    assert meta["capture_mode"] == SYNC_CAPTURE_MODE


def test_load_sync_calibration_rejects_unknown_offset_sign(tmp_path):
    path = tmp_path / "sync.json"
    payload = _sync_payload()
    payload["offset_definition"] = "audio_t_minus_force_t"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="offset_definition"):
        rec.load_sync_calibration(path)


@pytest.mark.parametrize(
    "updates",
    [
        {"n_matched": 3},
        {"offset_std_ms": 8.0},
        {"unique_matches": False},
        {"measurement_method": "hand_entered"},
    ],
)
def test_load_sync_calibration_rejects_low_quality_measurement(tmp_path, updates):
    payload = _sync_payload()
    payload.update(updates)
    path = tmp_path / "sync.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quality"):
        rec.load_sync_calibration(path)


def test_load_sync_calibration_rejects_missing_or_mismatched_capture_mode(tmp_path):
    payload = _sync_payload()
    payload.pop("capture_mode")
    path = tmp_path / "missing_mode.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="capture_mode"):
        rec.load_sync_calibration(path)

    path = tmp_path / "wrong_mode.json"; path.write_text(json.dumps(_sync_payload()))
    expected = {**SYNC_CAPTURE_MODE, "audio_channel": 0}
    with pytest.raises(ValueError, match="capture_mode.*audio_channel.*does not match"):
        rec.load_sync_calibration(path, expected_capture_mode=expected)


def test_sync_identity_matches_stable_usb_descriptors_across_port_and_index_remapping(tmp_path):
    path = tmp_path / "sync.json"; path.write_text(json.dumps(_sync_payload()))
    runtime = copy.deepcopy(SYNC_CAPTURE_MODE)
    runtime["hall"]["requested_port"] = "/dev/ttyACM0"
    runtime["hall"]["device"] = "/dev/ttyACM0"
    runtime["audio"]["portaudio_index"] = 7
    runtime["audio"]["device_name"] = "Scarlett Solo USB Audio (hw:7,0)"

    meta = rec.load_sync_calibration(path, expected_capture_mode=runtime)

    assert meta["runtime_capture_mode"] == runtime
    assert meta["identity_match"]["hall_usb"] == "matched"
    assert meta["identity_match"]["audio_usb"] == "matched"
    assert meta["identity_match"]["audio_portaudio_index"] == "remapped"


@pytest.mark.parametrize(
    ("component", "serial_field"), [("hall", "usb_serial"), ("audio", "usb_serial")],
)
def test_sync_identity_rejects_different_usb_serial(tmp_path, component, serial_field):
    path = tmp_path / "sync.json"; path.write_text(json.dumps(_sync_payload()))
    runtime = copy.deepcopy(SYNC_CAPTURE_MODE)
    runtime[component][serial_field] = "DIFFERENT-SERIAL"
    with pytest.raises(ValueError, match=f"sync capture_mode {component}.*usb_serial"):
        rec.load_sync_calibration(path, expected_capture_mode=runtime)


def test_arm_sync_identity_fails_closed_without_audio_usb_serial():
    hall = copy.deepcopy(SYNC_CAPTURE_MODE["hall"])
    audio = copy.deepcopy(SYNC_CAPTURE_MODE["audio"])
    audio.update(usb_serial=None, usb_vid=None, usb_pid=None,
                 usb_identity_source="unavailable_on_this_host")

    with pytest.raises(ValueError, match="arm-connected.*audio USB serial"):
        rec.build_sync_capture_mode(
            audio_device="Scarlett Solo USB", audio_channel=1,
            hall_port="/dev/ttyACM0", force_calibration_sha256=FORCE_SHA256,
            hall_resolver=lambda _: hall, audio_resolver=lambda _: audio,
            require_audio_usb_serial=True,
        )


def test_non_arm_sync_identity_preserves_unavailable_audio_usb_diagnostic():
    hall = copy.deepcopy(SYNC_CAPTURE_MODE["hall"])
    audio = copy.deepcopy(SYNC_CAPTURE_MODE["audio"])
    audio.update(usb_serial=None, usb_vid=None, usb_pid=None,
                 usb_identity_source="unavailable_on_this_host")

    mode = rec.build_sync_capture_mode(
        audio_device="Scarlett Solo USB", audio_channel=1,
        hall_port="/dev/ttyACM0", force_calibration_sha256=FORCE_SHA256,
        hall_resolver=lambda _: hall, audio_resolver=lambda _: audio,
        require_audio_usb_serial=False,
    )
    assert mode["audio"]["usb_serial"] is None
    assert mode["audio"]["usb_identity_source"] == "unavailable_on_this_host"


def test_arm_mode_requires_sync_and_camera_latency_calibrations(tmp_path):
    with pytest.raises(ValueError, match="sync.*camera"):
        rec.validate_timing_inputs("left", None, None)


def test_load_camera_latency_calibration_is_hashed_and_unapplied(tmp_path):
    path = tmp_path / "camera_latency.json"
    path.write_text(json.dumps(_camera_payload()))
    meta = rec.load_camera_latency_calibration(
        path, expected_capture_mode={"backend": "orbbec", "camera_id": "0"},
    )
    assert meta["latency_ms"] == pytest.approx(12.0)
    assert len(meta["calibration_sha256"]) == 64
    assert meta["applied"] is False
    assert meta["capture_mode"] == CAMERA_CAPTURE_MODE


@pytest.mark.parametrize(
    "updates",
    [
        {"latency_ms": 0.0},
        {"latency_std_ms": 8.0},
        {"n_samples": 5},
        {"measurement_method": "hand_entered"},
    ],
)
def test_load_camera_latency_rejects_low_quality_measurement(tmp_path, updates):
    payload = _camera_payload()
    payload.update(updates)
    path = tmp_path / "camera.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quality"):
        rec.load_camera_latency_calibration(path)


def test_load_camera_latency_rejects_missing_or_mismatched_capture_mode(tmp_path):
    payload = _camera_payload()
    payload.pop("capture_mode")
    path = tmp_path / "missing_mode.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="capture_mode"):
        rec.load_camera_latency_calibration(path)

    path = tmp_path / "wrong_mode.json"; path.write_text(json.dumps(_camera_payload()))
    with pytest.raises(ValueError, match="capture_mode.*backend.*does not match"):
        rec.load_camera_latency_calibration(
            path, expected_capture_mode={"backend": "avfoundation"},
        )


def test_camera_frame_dimensions_must_match_latency_calibration_mode():
    frame = SimpleNamespace(
        frame=SimpleNamespace(shape=(480, 640, 3)),
        time_source="orbbec_device_and_host_arrival",
        device_t_ms=123.0,
        depth=SimpleNamespace(ndim=2, shape=(480, 640)),
    )
    validation = rec.validate_camera_frame_mode(frame, {"capture_mode": CAMERA_CAPTURE_MODE})
    assert validation == {
        "width": "matched_frame_shape",
        "height": "matched_frame_shape",
        "fps": "not_verified_by_recorder",
        "time_source": "matched_frame_metadata",
        "device_timestamp": "present",
        "depth": "matched_pipeline_shape",
    }

    wrong = {"capture_mode": {**CAMERA_CAPTURE_MODE, "width": 1280}}
    with pytest.raises(rec.RecordingAborted, match="camera frame mode.*1280x480.*640x480"):
        rec.validate_camera_frame_mode(frame, wrong)


def test_camera_frame_mode_requires_matching_negotiated_fps_for_real_recording():
    frame = SimpleNamespace(
        frame=SimpleNamespace(shape=(480, 640, 3)),
        time_source="orbbec_device_and_host_arrival",
        device_t_ms=123.0,
        depth=SimpleNamespace(ndim=2, shape=(480, 640)),
    )
    stream = SimpleNamespace(
        negotiated_mode={
            "width": 640, "height": 480, "fps": 30.0, "color_format": "RGB",
            "depth_mode": {"width": 640, "height": 576, "fps": 30.0, "format": "Y16"},
            "align_mode": "software_depth_to_color",
            "frame_aggregate_mode": "full_frame_require",
        },
        device_identity="0",
    )
    validation = rec.validate_camera_frame_mode(
        frame, {"capture_mode": CAMERA_CAPTURE_MODE}, stream,
    )
    assert validation["fps"] == "matched_negotiated_mode"
    assert validation["camera_id"] == "matched_device_serial"
    assert validation["pipeline"] == "matched_negotiated_pipeline"
    assert validation["negotiated_mode"] == stream.negotiated_mode

    stream.negotiated_mode["fps"] = 15.0
    with pytest.raises(rec.RecordingAborted, match="negotiated FPS"):
        rec.validate_camera_frame_mode(frame, {"capture_mode": CAMERA_CAPTURE_MODE}, stream)


def test_video_container_uses_camera_fps_not_control_loop_rate():
    stream = SimpleNamespace(negotiated_mode={"fps": 30.0})
    timing = {"capture_mode": {"fps": 30.0}}
    assert rec.video_writer_fps(stream, timing, 50.0) == pytest.approx(30.0)


def test_force_calibration_is_parsed_hashed_and_snapshotted_from_one_read(
    monkeypatch, tmp_path,
):
    source = tmp_path / "mutable_force.json"
    bytes_a = b'{"calibrated": true, "version": "A"}\n'
    bytes_b = b'{"calibrated": true, "version": "B"}\n'
    source.write_bytes(bytes_a)
    source_reads = []
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path):
        if path.resolve() == source.resolve():
            source_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    parsed = object()

    class Parser:
        @classmethod
        def from_bytes(cls, raw):
            assert raw == bytes_a
            # Simulate the source being replaced A -> B -> A while validation is in progress.
            source.write_bytes(bytes_b)
            source.write_bytes(bytes_a)
            return parsed

    calibration, metadata = rec.load_force_calibration_snapshot(source, tmp_path / "session", Parser)

    snapshot = Path(metadata["calibration_path"])
    assert calibration is parsed
    assert len(source_reads) == 1
    assert snapshot.read_bytes() == bytes_a
    assert metadata["calibration_sha256"] == hashlib.sha256(bytes_a).hexdigest()
    assert metadata["source_calibration_path"] == str(source.resolve())


def test_camera_time_row_preserves_host_and_device_clocks():
    frame = SimpleNamespace(
        index=3, t=10.25, device_t_ms=1234.5,
        time_source="orbbec_device_and_host_arrival",
    )
    assert rec.camera_time_row(frame) == (
        3, 10.25, 1234.5, "orbbec_device_and_host_arrival",
    )


def test_observation_timestamp_is_sampled_after_camera_and_rejects_future_reference():
    frame = SimpleNamespace(t=10.0, frame=object())
    assert rec.observation_timestamp(frame, clock=lambda: 10.01) == pytest.approx(10.01)
    with pytest.raises(rec.RecordingAborted, match="future camera"):
        rec.observation_timestamp(frame, clock=lambda: 9.99)


def test_arm_mode_requires_sensor_first_ready_and_start_barriers(tmp_path):
    with pytest.raises(ValueError, match="ready.*start"):
        rec.validate_capture_barriers("left", None, None)
    rec.validate_capture_barriers("left", tmp_path / "ready.json", tmp_path / "start")


def test_readiness_requires_distinct_safe_force_samples_and_live_audio_camera():
    force = [SimpleNamespace(status="OK", t=float(i), device_t_ms=float(i)) for i in range(5)]
    audio = SimpleNamespace(status="OK")
    camera = SimpleNamespace(status="OK", frame=object())
    assert rec.readiness_failures(force, audio, camera) == []
    force[-1] = SimpleNamespace(status="SCREENED_LIMIT", t=5.0, device_t_ms=5.0)
    assert "force:SCREENED_LIMIT" in rec.readiness_failures(force, audio, camera)


@pytest.mark.parametrize(
    ("audio_status", "camera_status", "has_frame", "expected"),
    [
        ("OK", "OK", True, []),
        ("STALE", "OK", True, ["audio:STALE"]),
        ("NO_DATA", "OK", True, ["audio:NO_DATA"]),
        ("OK", "STALE", True, ["camera:STALE"]),
        ("OK", "NO_DATA", False, ["camera:NO_DATA"]),
    ],
)
def test_runtime_health_rejects_mid_capture_audio_or_camera_dropout(
    audio_status, camera_status, has_frame, expected,
):
    audio = SimpleNamespace(status=audio_status)
    camera = SimpleNamespace(status=camera_status, frame=object() if has_frame else None)
    assert rec.runtime_stream_failures(audio, camera) == expected


def test_runtime_health_requires_depth_for_orbbec_collection():
    audio = SimpleNamespace(status="OK")
    camera = SimpleNamespace(status="OK", frame=object(), depth=None)
    assert rec.runtime_stream_failures(audio, camera, require_depth=True) == ["camera:NO_DEPTH"]


def test_hard_limit_latch_is_latched_and_fail_closed(tmp_path):
    latch = rec.SafetyStop(tmp_path)
    latch(SimpleNamespace(force_n=5.6, status="HARD_LIMIT"))
    assert latch.tripped
    with pytest.raises(rec.RecordingAborted, match="hard limit.*5.600 N"):
        latch.raise_if_tripped()


def test_safety_latch_escalates_screened_event_to_hard_limit(tmp_path):
    latch = rec.SafetyStop(tmp_path)
    latch.observe(SimpleNamespace(force_n=4.7, status="SCREENED_LIMIT"))
    latch.observe(SimpleNamespace(force_n=5.6, status="HARD_LIMIT"))

    payload = json.loads((tmp_path / "SAFETY_ABORT.json").read_text())
    assert payload["severity"] == "hard"
    assert payload["force_status"] == "HARD_LIMIT"
    assert payload["force_n"] == pytest.approx(5.6)
    with pytest.raises(rec.RecordingAborted, match="hard limit.*5.600 N"):
        latch.raise_if_tripped()


@pytest.mark.parametrize(
    ("status", "severity"),
    [("SCREENED_LIMIT", "screened"), ("HARD_LIMIT", "hard"),
     ("STALE", "screened"), ("NO_DATA", "screened")],
)
def test_unsafe_force_status_writes_supervisor_sentinel(monkeypatch, tmp_path, status, severity):
    monkeypatch.setattr(rec.os, "kill", lambda *args: pytest.fail("recorder must not signal teleop"))
    stop = rec.SafetyStop(tmp_path)

    stop.observe(SimpleNamespace(force_n=5.6, status=status))
    stop.observe(SimpleNamespace(force_n=5.6, status=status))

    payload = json.loads((tmp_path / "SAFETY_ABORT.json").read_text())
    assert payload["force_status"] == status
    assert payload["severity"] == severity


@pytest.mark.parametrize("status", ["OK", "WARN"])
def test_safe_force_status_does_not_stop_teleop(monkeypatch, tmp_path, status):
    monkeypatch.setattr(rec.os, "kill", lambda *args: pytest.fail("recorder must not signal teleop"))
    stop = rec.SafetyStop(tmp_path)
    stop.observe(SimpleNamespace(force_n=4.0, status=status))
    assert not stop.tripped


def _install_hard_limit_fakes(monkeypatch):
    force_mod = ModuleType("force_sensor")

    class Calibration:
        calibrated = True
        max_screened_force_n = 4.6
        hard_limit_force_n = 5.5

        @classmethod
        def from_bytes(cls, raw):
            assert raw
            return cls()

    class Force:
        def __init__(self, port, calibration, *, log_csv, on_hard_limit):
            self.callback = on_hard_limit

        def __enter__(self):
            self.callback(SimpleNamespace(force_n=5.6, status="HARD_LIMIT"))
            return self

        def __exit__(self, *exc):
            return None

    force_mod.ForceCalibration = Calibration
    force_mod.ForceSensor = Force
    monkeypatch.setitem(sys.modules, "force_sensor", force_mod)

    audio_mod = ModuleType("audio_stream")

    class Audio:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    audio_mod.AudioStream = Audio
    monkeypatch.setitem(sys.modules, "audio_stream", audio_mod)

    camera_mod = ModuleType("camera_stream")

    class Camera(Audio):
        pass

    camera_mod.CameraStream = Camera
    monkeypatch.setitem(sys.modules, "camera_stream", camera_mod)


def test_record_session_marks_hard_limit_as_aborted(monkeypatch, tmp_path):
    _install_hard_limit_fakes(monkeypatch)
    monkeypatch.setattr(rec.time, "sleep", lambda _: None)
    session = tmp_path / "session"
    calibration = tmp_path / "commissioned-force.json"
    calibration.write_text("{}")

    with pytest.raises(rec.RecordingAborted, match="hard limit"):
        rec.record_session(
            str(tmp_path), 0.1, session_dir=str(session), hall_port="/dev/fake",
            hall_cal=str(calibration), allow_uncalibrated_force=False,
        )

    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["schema_version"] == "wam_session_v2"
    assert manifest["status"] == "aborted"
    assert manifest["training_eligible"] is False
    assert "hard limit" in manifest["abort_reason"]
    assert manifest["force"]["max_screened_force_n"] == pytest.approx(4.6)
    force_snapshot = Path(manifest["force"]["calibration_path"])
    assert force_snapshot.parent == session / "calibrations"
    assert force_snapshot.is_file()
    assert manifest["force"]["calibration_sha256"] == hashlib.sha256(
        force_snapshot.read_bytes()
    ).hexdigest()
