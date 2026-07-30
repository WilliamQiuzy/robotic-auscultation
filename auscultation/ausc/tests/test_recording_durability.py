"""Long-session and interrupted-session durability contracts for the formal recorder."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

AUSC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUSC))

import record_session as rec  # noqa: E402


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fp = self.path.open("xb")
        self.opened = True

    def isOpened(self):  # noqa: N802
        return self.opened

    def write(self, _frame):
        self.fp.write(b"F")
        self.fp.flush()

    def release(self):
        if self.opened:
            self.fp.close()
            self.opened = False


def _writer_factory(path, _fps, _size):
    return _FakeWriter(Path(path))


def _validator(path):
    return Path(path).read_bytes().count(b"F")


def _stitcher(paths, output, _fps, _size):
    raw = b"".join(Path(path).read_bytes() for path in paths)
    Path(output).write_bytes(raw)
    return raw.count(b"F")


def test_segmented_video_finalization_has_exact_count_hash_and_compatible_rgb_name(tmp_path):
    recorder = rec.SegmentedVideoRecorder(
        tmp_path, fps=30.0, size=(8, 6), segment_frames=2,
        writer_factory=_writer_factory, video_validator=_validator,
        video_stitcher=_stitcher,
    )
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    for index in range(5):
        recorder.write(frame, frame_index=index)

    result = recorder.finalize()

    assert result["frame_count"] == 5
    assert result["decoded_frame_count"] == 5
    assert result["segment_count"] == 3
    assert len(result["sha256"]) == 64
    assert (tmp_path / "rgb.mp4").read_bytes() == b"FFFFF"
    records = [
        json.loads(line)
        for line in (tmp_path / "rgb_segments.jsonl").read_text().splitlines()
    ]
    assert [record["frame_count"] for record in records] == [2, 2, 1]


def test_interrupted_video_recovers_only_sealed_prefix_and_never_production_seals(tmp_path):
    recorder = rec.SegmentedVideoRecorder(
        tmp_path, fps=30.0, size=(8, 6), segment_frames=2,
        writer_factory=_writer_factory, video_validator=_validator,
        video_stitcher=_stitcher,
    )
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    for index in range(5):
        recorder.write(frame, frame_index=index)
    active_partial = tmp_path / "rgb_segments" / "segment_000002.inprogress.mp4"
    assert active_partial.read_bytes() == b"F"
    original_timeline = "frame_index,host_arrival_t_s,device_t_ms,time_source\n" + "".join(
        f"{index},{10 + index / 30:.6f},{index},camera\n" for index in range(5)
    )
    (tmp_path / "rgb_times.csv").write_text(original_timeline)

    result = rec.recover_interrupted_session(
        tmp_path, video_validator=_validator, video_stitcher=_stitcher,
    )

    assert result["status"] == "diagnostic_recovered"
    assert result["training_eligible"] is False
    assert result["video"]["recovered_frame_count"] == 4
    assert (tmp_path / "rgb.recovered.mp4").read_bytes() == b"FFFF"
    assert active_partial.read_bytes() == b"F"
    assert (tmp_path / "rgb_times.csv").read_text() == original_timeline
    recovered_rows = (tmp_path / "rgb_times.recovered.csv").read_text().splitlines()
    assert len(recovered_rows) == 5
    assert recovered_rows[-1].startswith("3,")
    assert not (tmp_path / "rgb.mp4").exists()
    assert not (tmp_path / "FINALIZATION.json").exists()
    recovery = json.loads((tmp_path / "RECOVERY.json").read_text())
    assert recovery["training_eligible"] is False

    # Recovery is a resumable diagnostic transaction. Re-running after both
    # output commits must validate and reuse the exact same immutable outputs.
    second = rec.recover_interrupted_session(
        tmp_path, video_validator=_validator, video_stitcher=_stitcher,
    )
    assert second["video"]["sha256"] == result["video"]["sha256"]
    assert second["video"]["timeline"]["sha256"] == result["video"]["timeline"]["sha256"]


def test_interrupted_video_recovery_resumes_after_video_commit_before_timeline(tmp_path):
    recorder = rec.SegmentedVideoRecorder(
        tmp_path, fps=30.0, size=(8, 6), segment_frames=2,
        writer_factory=_writer_factory, video_validator=_validator,
        video_stitcher=_stitcher,
    )
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    for index in range(4):
        recorder.write(frame, frame_index=index)
    (tmp_path / "rgb_times.csv").write_text(
        "frame_index,host_arrival_t_s,device_t_ms,time_source\n"
        + "".join(f"{index},{10 + index / 30:.6f},{index},camera\n" for index in range(4))
    )
    # Model a process crash immediately after the recovered video rename.
    (tmp_path / "rgb.recovered.mp4").write_bytes(b"FFFF")

    result = rec.recover_interrupted_session(
        tmp_path, video_validator=_validator, video_stitcher=_stitcher,
    )

    assert result["video"]["recovered_frame_count"] == 4
    assert (tmp_path / "rgb.recovered.mp4").read_bytes() == b"FFFF"
    assert len((tmp_path / "rgb_times.recovered.csv").read_text().splitlines()) == 5


def test_depth_frame_atomic_write_requires_uint16_2d_and_exact_rgb_shape(tmp_path):
    destination = tmp_path / "depth.npy"
    with pytest.raises(rec.RecordingAborted, match="uint16"):
        rec._atomic_save_npy(
            destination, np.ones((6, 8), dtype=np.float32), expected_shape=(6, 8),
        )
    assert not destination.exists()

    with pytest.raises(rec.RecordingAborted, match="shape"):
        rec._atomic_save_npy(
            destination, np.ones((5, 8), dtype=np.uint16), expected_shape=(6, 8),
        )
    assert not destination.exists()

    values = np.arange(48, dtype=np.uint16).reshape(6, 8)
    rec._atomic_save_npy(destination, values, expected_shape=(6, 8))
    persisted = np.load(destination, allow_pickle=False)
    assert persisted.dtype == np.uint16
    assert persisted.shape == (6, 8)
    assert np.array_equal(persisted, values)


def test_segmented_video_itself_rejects_noncontiguous_camera_indexes(tmp_path):
    recorder = rec.SegmentedVideoRecorder(
        tmp_path, fps=30.0, size=(8, 6), segment_frames=2,
        writer_factory=_writer_factory, video_validator=_validator,
        video_stitcher=_stitcher,
    )
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    recorder.write(frame, frame_index=10)

    with pytest.raises(rec.RecordingAborted, match="camera frame index gap.*missing 1"):
        recorder.write(frame, frame_index=12)

    recorder.abort()
    assert not (tmp_path / "rgb.mp4").exists()


def test_durable_csv_is_incremental_and_does_not_retain_rows(tmp_path):
    path = tmp_path / "obs.csv"
    timeline = rec.DurableCsv(path, "a,b\n", fsync_interval_s=0.0)
    for index in range(10_000):
        timeline.append(f"{index},{index + 1}\n")
    timeline.close()

    assert timeline.count == 10_000
    assert not hasattr(timeline, "rows")
    assert path.read_text().splitlines()[-1] == "9999,10000"


def test_formal_record_session_uses_streaming_audio_segmented_video_and_disk_timelines(
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
        reads = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            type(self).reads += 1
            return SimpleNamespace(
                status="OK", force_n=(type(self).reads % 10) / 10.0,
                t=100.0 + type(self).reads / 100.0,
                device_t_ms=float(type(self).reads),
            )

    force_mod.ForceCalibration = Calibration
    force_mod.ForceSensor = Force
    monkeypatch.setitem(sys.modules, "force_sensor", force_mod)

    audio_mod = ModuleType("audio_stream")
    audio_calls = {"init": None, "begin": [], "end": []}

    class Audio:
        def __init__(self, *_args, **kwargs):
            audio_calls["init"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def level_db(self):
            return SimpleNamespace(status="OK", rms_db=-20.0, peak_db=-10.0)

        def begin_recording(self, session_dir, *, t0):
            audio_calls["begin"].append((Path(session_dir), t0))

        def end_recording(self, *, t_end):
            audio_calls["end"].append(t_end)
            path = audio_calls["begin"][0][0] / "audio.wav"
            path.write_bytes(b"fake-wav")
            t0 = audio_calls["begin"][0][1]
            return SimpleNamespace(
                path=str(path), sr=16_000, t_start=t0, t_end=t_end,
                n_samples=max(1, int((t_end - t0) * 16_000)), status="OK",
                peak_db=-10.0, sha256=rec._sha256_file(path),
            )

    audio_mod.AudioStream = Audio
    monkeypatch.setitem(sys.modules, "audio_stream", audio_mod)

    camera_mod = ModuleType("camera_stream")

    camera_instances = []

    class Camera:
        negotiated_mode = {"width": 8, "height": 6, "fps": 30.0}
        inject_gap = False

        def __init__(self, *_args, **_kwargs):
            camera_instances.append(self)
            self.index = 1
            self.delivery_calls = 0
            self.begin_calls = 0
            self.interval_id = 7
            self.drain_interval_ids = []
            self.end_interval_ids = []
            self.interval_active = False
            self.delivery_queue = []
            self.delivery_enqueued = 0
            self.delivery_delivered = 0
            self.first_enqueued = None
            self.last_enqueued = None
            self.last = self._make_frame(self.index, rec.time.monotonic())

        def _make_frame(self, index, host_t):
            return SimpleNamespace(
                status="OK", frame=np.zeros((6, 8, 3), dtype=np.uint8),
                index=index, t=host_t,
                device_t_ms=float(index), time_source="opencv_host_arrival",
                depth=None, depth_scale=None,
            )

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def latest_frame(self):
            if self.interval_active and not self.delivery_queue:
                self.delivery_calls += 1
                self.index += (
                    2 if type(self).inject_gap and self.delivery_calls == 1 else 1
                )
                self.last = self._make_frame(self.index, rec.time.monotonic())
                self.delivery_queue.append(self.last)
                self.delivery_enqueued += 1
                if self.first_enqueued is None:
                    self.first_enqueued = self.index
                self.last_enqueued = self.index
            return self.last

        def begin_delivery_interval(self, *, timeout_s=2.0):
            del timeout_s
            self.begin_calls += 1
            self.interval_active = True
            t_start = rec.time.monotonic()
            return SimpleNamespace(
                interval_id=self.interval_id,
                t_start=t_start,
                frame=self.last,
                proof={
                    "boundary_latched_atomically": True,
                    "interval_id": self.interval_id,
                    "start_boundary_frame_index": self.index,
                    "start_boundary_host_t": t_start,
                    "inflight_pre_boundary_seen": False,
                    "inflight_pre_boundary_completed": True,
                    "inflight_pre_boundary_produced_frame": False,
                    "capacity": 8,
                    "overflow_baseline": 0,
                },
            )

        def drain_delivery(self, max_frames=None, interval_id=None):
            self.drain_interval_ids.append(interval_id)
            limit = len(self.delivery_queue) if max_frames is None else int(max_frames)
            batch = tuple(self.delivery_queue[:limit])
            del self.delivery_queue[:limit]
            self.delivery_delivered += len(batch)
            return batch

        def delivery_health(self):
            return {
                "status": "OK", "interval_active": self.interval_active,
                "accepting": self.interval_active, "closing": False,
                "interval_id": self.interval_id,
                "capacity": 8, "queue_depth": len(self.delivery_queue),
                "max_queue_depth": 1,
                "interval_enqueued_frames": self.delivery_enqueued,
                "interval_delivered_frames": self.delivery_delivered,
                "first_enqueued_frame_index": self.first_enqueued,
                "last_enqueued_frame_index": self.last_enqueued,
                "last_delivered_frame_index": (
                    self.last_enqueued if self.delivery_delivered else None
                ),
                "interval_overflow_frames": 0, "overflow_frames": 0,
                "first_overflow_frame_index": None,
                "rejected_after_overflow": 0, "discarded_frames": 0,
            }

        def latch_end_boundary(self, *, timeout_s=2.0, interval_id=None):
            del timeout_s
            self.end_interval_ids.append(interval_id)
            t_end = rec.time.monotonic()
            self.index += 1
            self.last = self._make_frame(self.index, t_end)
            self.delivery_queue.append(self.last)
            self.delivery_enqueued += 1
            if self.first_enqueued is None:
                self.first_enqueued = self.index
            self.last_enqueued = self.index
            self.interval_active = False
            return SimpleNamespace(
                t_end=t_end,
                frame=self.last,
                proof={
                    "boundary_latched_atomically": True,
                    "inflight_pre_boundary_seen": True,
                    "inflight_pre_boundary_completed": True,
                    "inflight_pre_boundary_produced_frame": True,
                    "post_boundary_overwrite_prevented": True,
                    "interval_id": self.interval_id,
                    "delivery_capacity": 8,
                    "delivery_queue_depth": len(self.delivery_queue),
                    "delivery_enqueued_frames": self.delivery_enqueued,
                    "delivery_delivered_frames": self.delivery_delivered,
                    "delivery_first_frame_index": self.first_enqueued,
                    "delivery_last_frame_index": self.last_enqueued,
                    "delivery_overflow_frames": 0,
                    "snapshot_frame_index": self.index,
                    "snapshot_host_t": t_end,
                },
            )

        def abort_delivery_interval(self):
            self.interval_active = False
            self.delivery_queue.clear()
            return self.delivery_health()

    camera_mod.CameraStream = Camera
    monkeypatch.setitem(sys.modules, "camera_stream", camera_mod)

    cv2_mod = ModuleType("cv2")
    cv2_mod.COLOR_RGB2BGR = 1
    cv2_mod.cvtColor = lambda frame, _code: np.ascontiguousarray(frame[:, :, ::-1])
    cv2_mod.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

    video_instances = []

    class Video:
        def __init__(self, session_dir, *, fps, size, **_kwargs):
            self.session_dir = Path(session_dir)
            self.frames = []
            self.fps = fps
            self.size = size
            video_instances.append(self)

        def write(self, _frame, *, frame_index):
            self.frames.append(frame_index)

        def finalize(self):
            (self.session_dir / "rgb.mp4").write_bytes(b"video")
            return {
                "path": "rgb.mp4", "frame_count": len(self.frames),
                "decoded_frame_count": len(self.frames), "segment_count": 1,
                "segment_frames_limit": 150, "bytes": 5,
                "sha256": rec._sha256_file(self.session_dir / "rgb.mp4"),
            }

        def cleanup_segments(self):
            pass

        def abort(self):
            pass

    monkeypatch.setattr(rec, "SegmentedVideoRecorder", Video)
    decisions = iter([False, False, True])
    monkeypatch.setattr(rec, "capture_should_stop", lambda *_args, **_kwargs: next(decisions))
    clock = iter(100.5 + index / 100.0 for index in range(10_000))
    monkeypatch.setattr(rec.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rec.time, "sleep", lambda _seconds: None)

    calibration = tmp_path / "force.json"
    calibration.write_text("{}")
    session = tmp_path / "formal"
    result = rec.record_session(
        str(tmp_path), 3_600.0, session_dir=str(session), hall_port="/dev/fake",
        hall_cal=str(calibration), camera_backend="fake", storage_reserve_bytes=0,
        disk_check_interval_s=0.0, allow_missing_streams=True,
        code_release_id="code_release-" + "a" * 20,
        runtime_release_id="perception_runtime-" + "b" * 20,
        runtime_lock_sha256="c" * 64,
    )

    assert result == str(session)
    assert audio_calls["init"]["buffer_seconds"] <= 12.0
    assert len(audio_calls["begin"]) == len(audio_calls["end"]) == 1
    assert video_instances and len(video_instances[0].frames) == 3
    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["status"] == "sensors_complete"
    assert manifest["training_eligible"] is False
    assert manifest["capture_durability"]["video"]["decoded_frame_count"] == 3
    assert manifest["capture_durability"]["audio"]["streamed_to_disk"] is True
    assert manifest["capture_durability"]["capture_fifo"]["start_boundary_frame_index"] == 1
    assert manifest["capture_durability"]["capture_fifo"]["first_accepted_frame_index"] == 2
    assert manifest["capture_durability"]["capture_fifo"]["start_boundary_verified"] is True
    end_proof = manifest["capture_durability"]["camera_end_boundary"]
    assert end_proof["snapshot_frame_index"] == 4
    assert end_proof["tail_frame_submitted"] is False
    assert end_proof["tail_frame_already_submitted"] is True
    assert end_proof["fifo_last_submitted_frame_index"] == 4
    assert end_proof["fifo_last_processed_frame_index"] == 4
    assert end_proof["boundary_latched_atomically"] is True
    assert manifest["capture_durability"]["camera_delivery_fifo"]["queue_depth"] == 0
    assert manifest["capture_durability"]["camera_start_boundary"][
        "boundary_latched_atomically"
    ] is True
    assert manifest["release_provenance"] == {
        "schema_version": "wam_capture_release_provenance_v1",
        "code": {"release_id": "code_release-" + "a" * 20},
        "runtime": {
            "release_id": "perception_runtime-" + "b" * 20,
            "lock_sha256": "c" * 64,
        },
    }
    assert camera_instances[0].drain_interval_ids
    assert all(value == 7 for value in camera_instances[0].drain_interval_ids)
    assert camera_instances[0].end_interval_ids == [7]
    assert manifest["n_obs"] == 2
    assert len((session / "rgb_times.csv").read_text().splitlines()) == 4
    assert len((session / "obs.csv").read_text().splitlines()) == 3

    # Production integration gate: a one-slot CameraStream skip must abort the
    # formal session with durable missing-frame evidence, never seal a shorter
    # apparently valid rgb.mp4.
    Camera.inject_gap = True
    Force.reads = 0
    decisions = iter([False, False, True])
    monkeypatch.setattr(rec, "capture_should_stop", lambda *_args, **_kwargs: next(decisions))
    clock = iter(200.5 + index / 100.0 for index in range(10_000))
    monkeypatch.setattr(rec.time, "monotonic", lambda: next(clock))
    gap_session = tmp_path / "formal-gap"

    with pytest.raises(rec.RecordingAborted, match="camera Start boundary gap.*missing 1"):
        rec.record_session(
            str(tmp_path), 3_600.0, session_dir=str(gap_session), hall_port="/dev/fake",
            hall_cal=str(calibration), camera_backend="fake", storage_reserve_bytes=0,
            disk_check_interval_s=0.0, allow_missing_streams=True,
            code_release_id="code_release-" + "a" * 20,
            runtime_release_id="perception_runtime-" + "b" * 20,
            runtime_lock_sha256="c" * 64,
        )

    gap_manifest = json.loads((gap_session / "manifest.json").read_text())
    assert gap_manifest["status"] == "aborted"
    assert gap_manifest["training_eligible"] is False
    assert gap_manifest["capture_durability"]["capture_fifo"]["missing_camera_frames"] == 1
    assert gap_manifest["capture_durability"]["capture_fifo"]["start_boundary_frame_index"] == 1
    assert gap_manifest["capture_durability"]["capture_fifo"]["start_boundary_verified"] is False
    assert not (gap_session / "rgb.mp4").exists()
    assert not (gap_session / "FINALIZATION.json").exists()


def test_capture_fifo_rejects_camera_index_gap_and_reports_missing_count():
    processed = []
    writer = rec.BoundedCaptureWriter(
        lambda frame: processed.append(frame.index), capacity=2,
    ).start(t0=10.0)
    writer.submit(SimpleNamespace(index=41, t=10.1))

    with pytest.raises(rec.RecordingAborted, match="camera frame index gap.*missing 1"):
        writer.submit(SimpleNamespace(index=43, t=10.2))

    status = writer.status()
    assert status["accepted_frames"] == 1
    assert status["missing_camera_frames"] == 1
    assert status["last_submitted_frame_index"] == 41
    writer.abort()


def test_capture_fifo_verifies_first_frame_against_start_boundary():
    writer = rec.BoundedCaptureWriter(
        lambda _frame: None, capacity=2,
    ).start(
        t0=10.0, camera_boundary_index=100, camera_boundary_t=9.99,
        camera_backend="orbbec",
    )

    with pytest.raises(
        rec.RecordingAborted,
        match=r"camera Start boundary gap 100->102; missing 1",
    ):
        writer.submit(SimpleNamespace(index=102, t=10.01))

    status = writer.status()
    assert status["sequence_policy"] == "strict_contiguous_camera_index"
    assert status["camera_backend"] == "orbbec"
    assert status["start_boundary_frame_index"] == 100
    assert status["start_boundary_frame_t"] == pytest.approx(9.99)
    assert status["first_expected_frame_index"] == 101
    assert status["first_accepted_frame_index"] is None
    assert status["start_boundary_verified"] is False
    assert status["missing_camera_frames"] == 1
    writer.abort()


def test_capture_fifo_records_successful_first_frame_boundary_proof():
    writer = rec.BoundedCaptureWriter(
        lambda _frame: None, capacity=2,
    ).start(
        t0=10.0, camera_boundary_index=100, camera_boundary_t=9.99,
        camera_backend="orbbec",
    )
    writer.submit(SimpleNamespace(index=101, t=10.01))

    status = writer.finish(t_end=10.02)

    assert status["first_accepted_frame_index"] == 101
    assert status["start_boundary_verified"] is True
    assert status["missing_camera_frames"] == 0


def test_capture_fifo_overflow_is_fail_closed_and_never_overwrites():
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def process(frame):
        processed.append(frame.index)
        if frame.index == 1:
            entered.set()
            assert release.wait(1.0)

    writer = rec.BoundedCaptureWriter(process, capacity=1).start(t0=10.0)
    writer.submit(SimpleNamespace(index=1, t=10.1))
    assert entered.wait(1.0)
    writer.submit(SimpleNamespace(index=2, t=10.2))

    with pytest.raises(rec.RecordingAborted, match="capture writer FIFO overflow"):
        writer.submit(SimpleNamespace(index=3, t=10.3))

    assert writer.status()["overflow_frames"] == 1
    release.set()
    writer.abort()
    assert processed == [1, 2]


def test_capture_fifo_rejects_frame_from_before_start_boundary():
    writer = rec.BoundedCaptureWriter(lambda _frame: None, capacity=2).start(t0=10.0)
    with pytest.raises(rec.RecordingAborted, match="predates Start boundary"):
        writer.submit(SimpleNamespace(index=1, t=9.999))
    assert writer.status()["boundary_rejections"] == 1
    writer.abort()


def test_capture_fifo_worker_error_and_end_drain_are_fail_closed():
    failed = rec.BoundedCaptureWriter(
        lambda frame: (_ for _ in ()).throw(OSError(f"disk failed at {frame.index}")),
        capacity=2,
    ).start(t0=10.0)
    failed.submit(SimpleNamespace(index=1, t=10.1))
    with pytest.raises(rec.RecordingAborted, match="capture writer failed.*disk failed"):
        failed.finish(t_end=10.2)
    assert "disk failed" in failed.status()["worker_error"]

    entered = threading.Event()
    release = threading.Event()

    def slow(_frame):
        entered.set()
        assert release.wait(1.0)

    draining = rec.BoundedCaptureWriter(slow, capacity=2).start(t0=20.0)
    draining.submit(SimpleNamespace(index=8, t=20.1))
    assert entered.wait(1.0)
    outcome = {}
    thread = threading.Thread(
        target=lambda: outcome.setdefault("status", draining.finish(t_end=20.2)), daemon=True,
    )
    thread.start()
    assert thread.is_alive(), "End must wait for every accepted frame"
    release.set()
    thread.join(1.0)
    assert outcome["status"]["accepted_frames"] == 1
    assert outcome["status"]["processed_frames"] == 1
    assert outcome["status"]["drained"] is True
