"""Durability contracts for diagnostic-only camera/perception capture."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import perception_capture  # noqa: E402
from auscultation.ausc.audio.audio_stream import (  # noqa: E402
    AudioSignalAuthenticityAccumulator,
)
from auscultation.ausc import realtime_q  # noqa: E402

FORCE_FEEDBACK = SCRIPTS.parent / "mechanical_sensor" / "force_feedback"
sys.path.insert(0, str(FORCE_FEEDBACK))
import force_capture  # noqa: E402
import force_sensor  # noqa: E402

DiagnosticPerceptionCapture = perception_capture.DiagnosticPerceptionCapture
ProductionTeleopCapture = perception_capture.ProductionTeleopCapture


PROVENANCE = {
    "camera": {"serial": "CL8326300A0", "mode": "640x576-rgbd-30"},
    "model": {
        "release_id": "perception-model-test",
        "checkpoint_sha256": "a" * 64,
        "base_model_sha256": "b" * 64,
        "landmark_order_sha256": "c" * 64,
    },
    "runtime": {"code_release_id": "code-test", "lock_sha256": "d" * 64},
}


def _frame(index: int, *, value: int = 20):
    rgb = np.full((8, 12, 3), value, dtype=np.uint8)
    depth = np.full((8, 12), 900 + index, dtype=np.uint16)
    return SimpleNamespace(
        frame=rgb,
        depth=depth,
        depth_scale=1.0,
        t=100.0 + index / 30.0,
        index=index,
        status="OK",
        device_t_ms=2000.0 + index * 33.0,
        time_source="orbbec_device_and_host_arrival",
    )


def _tracking(frame) -> dict:  # noqa: ANN001
    return {
        "back_present": True,
        "initialized": True,
        "reason": "tracked",
        "frame_index": frame.index,
        "source_tracking_frame_index": frame.index,
        "source_tracking_host_t_s": frame.t,
        "source_tracking_device_t_ms": frame.device_t_ms,
        "tracking_association": "same_camera_frame",
        "tracking_age_s": 0.0,
        "host_t": frame.t,
        "device_t": frame.device_t_ms,
        "points": [{
            "name": "R1",
            "pixel": [4.0, 5.0],
            "xyz_camera": [0.1, 0.2, 0.9],
            "status": "TRACKING",
            "measured": True,
            "position_std": 0.002,
        }],
    }


def _write_valid_audio_artifacts(tmp_path: Path) -> dict:
    """Publish the exact semantic shape emitted by AudioRecordingCapture."""

    sample_rate = 16_000
    input_sample_rate = 44_100
    n_samples = 1_600
    input_samples = 4_410
    t_start = 10.0
    t_end = 10.1
    pcm = (((np.arange(n_samples) % 257) - 128) * 32).astype("<i2")
    authenticity_analyzer = AudioSignalAuthenticityAccumulator(sample_rate)
    authenticity_analyzer.update(pcm)
    signal_authenticity = authenticity_analyzer.finish()
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    sha = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    (tmp_path / "audio_times.jsonl").write_text(json.dumps({
        "sample_offset": 0,
        "sample_count": n_samples,
        "t_start": t_start,
        "t_end": t_end,
        "source_input_sample_count": input_samples,
        "input_sample_rate": input_sample_rate,
        "sample_rate": sample_rate,
    }, sort_keys=True) + "\n")
    (tmp_path / "audio_input_times.jsonl").write_text(json.dumps({
        "input_sample_offset": 0,
        "input_sample_count": input_samples,
        "t_start": t_start,
        "t_end": t_end,
        "source_block_start_t": t_start,
        "source_block_end_t": t_end,
    }, sort_keys=True) + "\n")
    (tmp_path / "audio_capture.json").write_text(json.dumps({
        "schema_version": "wam_audio_stream_capture_v1",
        "status": "sensors_finalized",
        "training_eligible": False,
        "sample_rate": sample_rate,
        "input_sample_rate": input_sample_rate,
        "partial_sample_rate": input_sample_rate,
        "partial_encoding": "signed_pcm_s16le_mono",
        "t0_monotonic": t_start,
        "audio_t0_monotonic": t_start,
        "audio_t_end_monotonic": t_end,
        "partial_n_samples": input_samples,
        "n_samples": n_samples,
        "sha256": sha,
        "signal_authenticity": signal_authenticity,
    }))
    return {
        "path": "audio.wav",
        "sample_rate": sample_rate,
        "input_sample_rate": input_sample_rate,
        "n_samples": n_samples,
        "t_start": t_start,
        "t_end": t_end,
        "status": "OK",
        "peak_db": 20.0 * np.log10(np.max(np.abs(pcm.astype(np.int32))) / 32_767),
        "sha256": sha,
    }


def _bind_current_audio_sha(tmp_path: Path, summary: dict) -> None:
    sha = hashlib.sha256((tmp_path / "audio.wav").read_bytes()).hexdigest()
    summary["sha256"] = sha
    metadata_path = tmp_path / "audio_capture.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = sha
    metadata_path.write_text(json.dumps(metadata))


def _write_valid_force_artifacts(tmp_path: Path) -> dict:
    recorder = force_capture.ForceCaptureRecorder(
        tmp_path,
        t0=10.0,
        calibration_identity={
            "sha256": "e" * 64,
            "calibrated": True,
            "source_name": "hall-calibration.json",
        },
        device_identity={
            "port": "/dev/serial/by-id/usb-RP2040-test",
            "baud": 115200,
            "usb_vid": "2e8a",
            "usb_pid": "0005",
            "serial_number": "53032847188a2e1c",
        },
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    )
    recorder.start()
    for index in range(501):
        raw = 31_000 + (index % 101)
        recorder.submit(force_sensor.ForceSample(
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
    summary = recorder.finish(t_end=15.0)
    return {
        "path": "force.csv",
        "t_start": summary.t_start,
        "t_end": summary.t_end,
        "n_samples": summary.n_samples,
        "status": summary.status,
        "sha256": summary.sha256,
        "device_t_start_ms": summary.device_t_start_ms,
        "device_t_end_ms": summary.device_t_end_ms,
        "calibration_sha256": summary.calibration_sha256,
        "device_identity_sha256": summary.device_identity_sha256,
    }


def _write_pinned_force_artifacts(tmp_path: Path) -> dict:
    calibration_payload = json.loads(
        (FORCE_FEEDBACK / "force_feedback_calibration_veo6.json").read_text(
            encoding="utf-8"
        )
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
        "source_name": "hall-calibration.json",
        "artifact_schema_version": "wam_force_calibration_json_v1",
        "release_id": f"hall-calibration-test-{calibration_sha256[:20]}",
    }
    calibration = force_sensor.ForceCalibration.from_bytes(calibration_raw)
    recorder = force_capture.ForceCaptureRecorder(
        tmp_path,
        t0=10.0,
        calibration_identity=calibration_identity,
        calibration_artifact=calibration_raw,
        device_identity={
            "port": "/dev/serial/by-id/usb-RP2040-test",
            "baud": 115200,
            "usb_vid": "2e8a",
            "usb_pid": "0005",
            "serial_number": "53032847188a2e1c",
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
    summary = recorder.finish(t_end=15.0)
    return {
        "path": "force.csv",
        "t_start": summary.t_start,
        "t_end": summary.t_end,
        "n_samples": summary.n_samples,
        "status": summary.status,
        "sha256": summary.sha256,
        "device_t_start_ms": summary.device_t_start_ms,
        "device_t_end_ms": summary.device_t_end_ms,
        "calibration_sha256": summary.calibration_sha256,
        "device_identity_sha256": summary.device_identity_sha256,
    }


def _write_quality_bound_force_artifacts(tmp_path: Path) -> dict:
    recorder = force_capture.ForceCaptureRecorder(
        tmp_path,
        t0=7.2,
        calibration_identity={
            "sha256": "e" * 64,
            "calibrated": True,
            "source_name": "hall-calibration.json",
        },
        device_identity={
            "port": "/dev/serial/by-id/usb-RP2040-quality-test",
            "baud": 115200,
            "usb_vid": "2e8a",
            "usb_pid": "0005",
            "serial_number": "quality-test",
        },
        max_device_gap_ms=400.0,
        max_host_gap_ms=400.0,
    )
    recorder.start()
    for index, inference_t in enumerate((7.25, 7.55, 7.85)):
        recorder.submit(force_sensor.ForceSample(
            t=inference_t - 0.05,
            raw=31_000 + index,
            hall_v=1.56 + index * 0.001,
            displacement_mm=1.20 + index * 0.01,
            force_n=4.0,
            status="OK",
            device_t_ms=100.0 + index * 300.0,
        ))
    summary = recorder.finish(t_end=7.8)
    return {
        "path": "force.csv",
        "t_start": summary.t_start,
        "t_end": summary.t_end,
        "n_samples": summary.n_samples,
        "status": summary.status,
        "sha256": summary.sha256,
        "device_t_start_ms": summary.device_t_start_ms,
        "device_t_end_ms": summary.device_t_end_ms,
        "calibration_sha256": summary.calibration_sha256,
        "device_identity_sha256": summary.device_identity_sha256,
    }


def _write_valid_quality_artifacts(tmp_path: Path) -> tuple[dict, dict, dict]:
    model_sha256 = "a" * 64
    calibration_sha256 = "c" * 64
    policy = {
        "schema_version": "wam_quality_policy_v1",
        "policy_id": "clinical-q-capture-test-v1",
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
    force_summary = _write_quality_bound_force_artifacts(tmp_path)
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = realtime_q.StreamingQuality(
        scorer=lambda samples, _sr: float(np.mean(samples)),
        mode="production",
        model_release_id=f"clinical-q-capture-test-{model_sha256[:20]}",
        model_sha256=model_sha256,
        policy=policy,
        packet_log_writer=writer,
    )
    for sequence, value, inference_t in (
        (30, 0.1, 7.25),
        (31, 0.2, 7.55),
        (32, 0.3, 7.85),
    ):
        samples = np.linspace(value, value + 0.01, 4_000, dtype=np.float32)
        # 4_000 window samples @ 16 kHz = 0.25 s = 12_000 source samples @ 48 kHz.
        # Production Q now binds every window to its exact source PCM16 range.
        window_index = sequence - 30
        quality.update(
            audio_window=SimpleNamespace(
                samples=samples,
                sr=16_000,
                t_start=inference_t - 0.25,
                t_end=inference_t,
                status="OK",
                raw_source={
                    "source_channel": 1,
                    "source_sample_start": window_index * 12_000,
                    "source_sample_end": (window_index + 1) * 12_000,
                    "source_sample_rate": 48_000,
                    "resampler_identity": {
                        "algorithm": "polyphase_fir_v1",
                        "input_sample_rate": 48_000,
                        "output_sample_rate": 16_000,
                    },
                },
            ),
            force_sample=SimpleNamespace(
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
    quality_authenticity = quality.quality_authenticity(now_monotonic_s=7.9)
    log_receipt = writer.finalize(
        quality_authenticity,
        now_monotonic_s=7.9,
    )
    return (
        realtime_q.quality_packet_log_summary(log_receipt),
        log_receipt,
        force_summary,
    )


class _Writer:
    def __init__(self, path: Path, *, opened: bool = True) -> None:
        self.path = Path(path)
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False
        if opened:
            self.path.write_bytes(b"FAKE-MP4-HEADER")

    def isOpened(self) -> bool:  # OpenCV-compatible spelling
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        if self.released:
            raise RuntimeError("write after release")
        self.frames.append(frame.copy())
        with self.path.open("ab") as fp:
            fp.write(hashlib.sha256(frame.tobytes()).digest())

    def release(self) -> None:
        if self.opened and not self.released:
            with self.path.open("ab") as fp:
                fp.write(b"FAKE-MP4-TRAILER")
        self.released = True


def _capture(
    tmp_path: Path, *, opened=True, free_bytes=10**9, validator=None,
    provenance=None, video_mode_probe=None, **kwargs,
):
    writers: list[_Writer] = []
    writer_by_path: dict[Path, _Writer] = {}

    def factory(path: Path, fps: float, size: tuple[int, int]):
        # OpenCV/FFmpeg chooses the container from the final suffix.  A name
        # ending in `.inprogress` silently fails on the real Linbox runtime.
        assert path.suffix == ".mp4"
        assert fps == 30.0
        assert size == (12, 8)
        writer = _Writer(path, opened=opened)
        writers.append(writer)
        writer_by_path[path] = writer
        return writer

    validated: list[Path] = []
    validation_threads: list[str] = []

    def writer_for(path: Path) -> _Writer | None:
        writer = writer_by_path.get(path)
        if writer is not None:
            return writer
        if _segment_name := path.name.removesuffix(".mp4"):
            return writer_by_path.get(path.with_name(_segment_name + ".inprogress.mp4"))
        return None

    def validate(path: Path) -> int:
        validated.append(path)
        validation_threads.append(threading.current_thread().name)
        if validator is not None and path.name == "rgb.mp4":
            return validator(path)
        writer = writer_for(path)
        if writer is not None:
            return len(writer.frames)
        prefix = b"FAKE-STITCHED-MP4:"
        raw = path.read_bytes()
        assert raw.startswith(prefix)
        return int(raw[len(prefix):])

    def stitch(paths, output: Path, fps: float, size: tuple[int, int]) -> int:
        assert fps == 30.0
        assert size == (12, 8)
        count = sum(len(writer_for(Path(path)).frames) for path in paths)
        output.write_bytes(b"FAKE-STITCHED-MP4:" + str(count).encode())
        return count

    require_robot = kwargs.pop("require_robot", False)
    require_force = kwargs.pop("require_force", False)
    capture_cls = kwargs.pop("capture_cls", None) or DiagnosticPerceptionCapture
    capture = capture_cls(
        tmp_path,
        fps=30.0,
        provenance=PROVENANCE if provenance is None else provenance,
        writer_factory=factory,
        video_validator=validate,
        video_mode_probe=video_mode_probe,
        video_stitcher=stitch,
        disk_usage=lambda _: shutil._ntuple_diskusage(2 * 10**9, 10**9, free_bytes),
        min_free_bytes=1024,
        depth_chunk_frames=2,
        checkpoint_every_frames=1,
        clock=lambda: 700.0,
        wall_time_ns=lambda: 1_721_234_567_000_000_123,
        require_robot=require_robot,
        require_force=require_force,
        **kwargs,
    )
    capture._test_validation_threads = validation_threads
    return capture, writers, validated


def test_production_teleop_capture_is_raw_and_never_self_promotes():
    # #6 aligned: the production writer is a thin subclass that reuses the diagnostic
    # machinery and seals the SAME raw diagnostic-shape completion (so the reconciler
    # recognises it). It differs ONLY in the manifest identity that marks it promotable;
    # it NEVER self-promotes -- the trusted finalizer is the sole promoter.
    assert issubclass(ProductionTeleopCapture, DiagnosticPerceptionCapture)
    assert ProductionTeleopCapture._COMPLETION_STATUS == "diagnostic_complete"
    assert ProductionTeleopCapture._SEAL_FILENAME == "DIAGNOSTIC_FINALIZATION.json"
    assert ProductionTeleopCapture._MANIFEST_CAPTURE_KIND == "production_teleop_v4"
    assert ProductionTeleopCapture._MANIFEST_SCHEMA_VERSION == "wam_session_v2"
    assert (
        ProductionTeleopCapture._MANIFEST_PROMOTION_CONTRACT_VERSION == "wam_promotion_v2"
    )


def test_production_manifest_is_a_raw_fail_closed_promotable_session(tmp_path):
    # #6 aligned: a production Start manifest satisfies the finalizer's _manifest_check
    # for a RAW promotable session -- schema_version=wam_session_v2, a valid session_id,
    # rehearsal_overrides=[], the promotion identity, AND training_eligible=False (the
    # writer never self-promotes; the finalizer is the sole promoter).
    capture, _w, _v = _capture(tmp_path, capture_cls=ProductionTeleopCapture)
    capture.start()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "wam_session_v2"
    assert manifest["capture_kind"] == "production_teleop_v4"
    assert manifest["promotion_contract_version"] == "wam_promotion_v2"
    assert manifest["session_id"] == tmp_path.name
    assert manifest["rehearsal_overrides"] == []
    assert manifest["training_eligible"] is False   # raw -- finalizer promotes


def test_diagnostic_manifest_has_no_promotion_identity(tmp_path):
    # A diagnostic Start manifest carries NO promotion fields (converter/finalizer
    # reject a diagnostic manifest, and _write_manifest forbids them).
    capture, _w, _v = _capture(tmp_path)
    capture.start()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "capture_kind" not in manifest
    assert "promotion_contract_version" not in manifest
    assert manifest["capture_mode"] == "diagnostic_perception"
    assert manifest["schema_version"] == "wam_diagnostic_perception_v1"


def test_missing_force_marks_channel_invalid_not_a_failure(tmp_path):
    # P0-4: force is advisory -- a missing force channel is NOT a failure; it only marks
    # contact_force_valid=False. The session still seals (video/other data preserved).
    capture, _, _ = _capture(tmp_path, require_force=True)
    capture.start()
    assert capture._validate_force_artifacts() is None   # no raise on missing force
    assert capture._contact_force_valid is False


def test_required_quality_packet_log_is_fail_closed_when_missing(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))

    with pytest.raises(RuntimeError, match="required quality packet log.*missing"):
        capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False


def test_unbound_optional_quality_artifacts_abort_instead_of_being_ignored(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_quality=False)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    _, _, force_summary = _write_valid_quality_artifacts(tmp_path)
    capture.set_force_summary(force_summary)

    with pytest.raises(RuntimeError, match="unbound quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_unbound_quality_staging_log_aborts_instead_of_being_ignored(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=False)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    (tmp_path / ".quality_inference_packets.jsonl.inprogress").write_text(
        '{"partial":true}\n'
    )

    with pytest.raises(RuntimeError, match="unbound quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_unbound_quality_model_input_staging_aborts_instead_of_being_ignored(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_quality=False)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    (tmp_path / ".quality_model_inputs.inprogress").mkdir()

    with pytest.raises(RuntimeError, match="unbound quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_unbound_optional_force_artifacts_abort_instead_of_being_ignored(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_force=False)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    _write_valid_force_artifacts(tmp_path)

    with pytest.raises(RuntimeError, match="unbound Hall-force"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_unbound_force_staging_artifact_aborts_instead_of_being_ignored(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_force=False)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    (tmp_path / "force.csv.inprogress").write_text(
        "host_t_s,raw,hall_v,displacement_mm,force_n,status,device_t_ms\n"
    )

    with pytest.raises(RuntimeError, match="unbound Hall-force"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_force_sidecar_is_verified_and_sealed_in_diagnostic_inventory(tmp_path):
    provenance = json.loads(json.dumps(PROVENANCE))
    provenance["force"] = {
        "calibration_sha256": "e" * 64,
        "sync_calibration_sha256": "f" * 64,
        "sync": {
            "offset_definition": "force_t_minus_audio_t",
            "audio_to_force_offset_ms": 1.25,
            "offset_std_ms": 0.2,
            "identity_match": {"hall_usb": "matched", "audio_usb": "matched"},
        },
    }
    capture, _, _ = _capture(
        tmp_path, require_force=True, provenance=provenance,
    )
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    force_summary = _write_valid_force_artifacts(tmp_path)
    capture.set_force_summary(force_summary)

    result = capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    force_metadata = json.loads((tmp_path / "force_capture.json").read_text())
    authenticity = force_metadata["signal_authenticity"]
    required = {"force.csv", "force_capture.json", "FORCE_CAPTURE_COMPLETE.json"}
    assert required <= manifest["artifacts"].keys()
    assert required <= seal["artifacts"].keys()
    assert manifest["force"] == force_summary
    assert seal["force"] == force_summary
    assert manifest["force_authenticity"] == authenticity
    assert seal["force_authenticity"] == authenticity
    assert manifest["provenance"]["force"] == provenance["force"]
    assert seal["provenance"]["force"] == provenance["force"]
    assert result["force"] == force_summary
    assert result["force_authenticity"] == authenticity
    assert result["training_eligible"] is False


def test_short_force_capture_marks_channel_invalid_not_a_failure(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_force=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    recorder = force_capture.ForceCaptureRecorder(
        tmp_path,
        t0=10.0,
        calibration_identity={
            "sha256": "e" * 64,
            "calibrated": True,
            "source_name": "hall-calibration.json",
        },
        device_identity={
            "port": "/dev/serial/by-id/usb-RP2040-test",
            "baud": 115200,
            "usb_vid": "2e8a",
            "usb_pid": "0005",
            "serial_number": "53032847188a2e1c",
        },
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    )
    recorder.start()
    for index in range(51):
        recorder.submit(force_sensor.ForceSample(
            t=10.0 + index * 0.01,
            raw=31_000,
            hall_v=1.561,
            displacement_mm=1.25,
            force_n=2.75,
            status="WARN",
            device_t_ms=100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()
    summary = recorder.finish(t_end=10.5)
    force_summary = {
        "path": "force.csv",
        "t_start": summary.t_start,
        "t_end": summary.t_end,
        "n_samples": summary.n_samples,
        "status": summary.status,
        "sha256": summary.sha256,
        "device_t_start_ms": summary.device_t_start_ms,
        "device_t_end_ms": summary.device_t_end_ms,
        "calibration_sha256": summary.calibration_sha256,
        "device_identity_sha256": summary.device_identity_sha256,
    }

    # P0-4: advisory -- low force authenticity (a short capture) marks the contact-force
    # channel invalid; it does NOT fail the session.
    capture.set_force_summary(force_summary)
    assert capture._contact_force_valid is False


def test_pinned_force_calibration_is_bound_into_manifest_and_seal(tmp_path):
    capture, _, _ = _capture(tmp_path, require_force=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    force_summary = _write_pinned_force_artifacts(tmp_path)
    capture.set_force_summary(force_summary)

    result = capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    expected_calibration = {
        "bytes": (tmp_path / "force_calibration.json").stat().st_size,
        "sha256": hashlib.sha256(
            (tmp_path / "force_calibration.json").read_bytes()
        ).hexdigest(),
    }
    assert manifest["artifacts"]["force_calibration.json"] == expected_calibration
    assert seal["artifacts"]["force_calibration.json"] == expected_calibration
    assert result["force_authenticity"]["promotion_eligible"] is True
    assert (
        result["force_authenticity"]["checks"]["calibration_recomputation"]
        == "PASS"
    )


def test_pinned_force_calibration_tamper_after_binding_aborts_finalization(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path, require_force=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    capture.set_force_summary(_write_pinned_force_artifacts(tmp_path))
    with (tmp_path / "force_calibration.json").open("ab") as handle:
        handle.write(b"tamper\n")

    with pytest.raises(RuntimeError, match="Hall-force"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_quality_packet_log_is_fully_verified_and_sealed(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    quality_summary, log_receipt, force_summary = (
        _write_valid_quality_artifacts(tmp_path)
    )
    capture.set_force_summary(force_summary)
    capture.set_quality_summary(quality_summary)

    result = capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    required = {
        "quality_inference_packets.jsonl",
        "quality_inference_authenticity.json",
        "quality_model_inputs/input_00000000000000000030.f32le",
        "quality_model_inputs/input_00000000000000000031.f32le",
        "quality_model_inputs/input_00000000000000000032.f32le",
    }
    assert required <= manifest["artifacts"].keys()
    assert required <= seal["artifacts"].keys()
    assert manifest["quality"] == quality_summary
    assert seal["quality"] == quality_summary
    assert manifest["quality_authenticity"] == log_receipt
    assert seal["quality_authenticity"] == log_receipt
    assert result["quality"] == quality_summary
    assert result["quality_authenticity"] == log_receipt
    assert result["training_eligible"] is False


def test_quality_packet_log_cannot_bind_without_verified_force_summary(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    quality_summary, _, _ = _write_valid_quality_artifacts(tmp_path)

    with pytest.raises(RuntimeError, match="verified Hall-force"):
        capture.set_quality_summary(quality_summary)

    assert capture._quality_summary is None


def test_quality_packet_log_tamper_after_binding_aborts_finalization(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    quality_summary, _, force_summary = _write_valid_quality_artifacts(tmp_path)
    capture.set_force_summary(force_summary)
    capture.set_quality_summary(quality_summary)
    with (tmp_path / "quality_inference_packets.jsonl").open("ab") as handle:
        handle.write(b'{"tampered":true}\n')

    with pytest.raises(RuntimeError, match="quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert not (tmp_path / "quality_model_inputs").exists()
    assert list(tmp_path.glob("quality_model_inputs.aborted*"))


def test_quality_model_input_tamper_after_binding_aborts_finalization(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    quality_summary, _, force_summary = _write_valid_quality_artifacts(tmp_path)
    capture.set_force_summary(force_summary)
    capture.set_quality_summary(quality_summary)
    model_input = (
        tmp_path
        / "quality_model_inputs"
        / "input_00000000000000000031.f32le"
    )
    with model_input.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00\x00\x00\x00")

    with pytest.raises(RuntimeError, match="quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_bound_quality_log_with_reappeared_staging_file_cannot_finalize(tmp_path):
    capture, _, _ = _capture(tmp_path, require_quality=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    quality_summary, _, force_summary = _write_valid_quality_artifacts(tmp_path)
    capture.set_force_summary(force_summary)
    capture.set_quality_summary(quality_summary)
    (tmp_path / ".quality_inference_packets.jsonl.inprogress").write_text(
        '{"late_partial":true}\n'
    )

    with pytest.raises(RuntimeError, match="quality packet log"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_force_sidecar_tamper_after_binding_aborts_finalization(tmp_path):
    capture, _, _ = _capture(tmp_path, require_force=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    capture.set_force_summary(_write_valid_force_artifacts(tmp_path))
    with (tmp_path / "force.csv").open("ab") as handle:
        handle.write(b"tamper\n")

    with pytest.raises(RuntimeError, match="Hall-force"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_required_robot_sidecar_is_fail_closed_when_missing(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))

    with pytest.raises(RuntimeError, match="required robot.*missing"):
        capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_start_rejects_stale_final_robot_commit_marker(tmp_path):
    (tmp_path / "robot_capture_summary.json").write_text("{}\n")
    capture, _, _ = _capture(tmp_path, require_robot=True)

    with pytest.raises(FileExistsError, match="robot_capture_summary.json"):
        capture.start()

    assert not (tmp_path / "manifest.json").exists()


def _robot_summary(**changes) -> dict:
    summary = {
        "schema": "wam_openarm_exact_control_capture_summary_v3",
        "status": "exact_control_capture_validated_socketcan_tx_enqueue",
        "side": "left",
        "controller_mode": "unilateral",
        "leader_can_interface": "can1",
        "follower_can_interface": "can3",
        "joint_names": [f"openarm_left_joint{index}" for index in range(1, 8)],
        "controller_pid": 4242,
        "controller_start_token": "proc-start-991",
        "publisher_identity": (
            f"pid=4242;start=proc-start-991;mode=unilateral;leader_can=can1;"
            f"follower_can=can3;urdf={'a' * 64};mapper={'b' * 64};"
            f"pair={'e' * 64};binary={'f' * 64};source={'1' * 64};"
            f"config={'2' * 64};sdk={'3' * 64}"
        ),
        "urdf_sha256": "a" * 64,
        "joint_mapper_sha256": "b" * 64,
        "camera_profile_sha256": "c" * 64,
        "timing_artifact_sha256": "d" * 64,
        "pair_contract_sha256": "e" * 64,
        "controller_binary_sha256": "f" * 64,
        "controller_source_manifest_sha256": "1" * 64,
        "follower_config_sha256": "2" * 64,
        "openarm_sdk_sha256": "3" * 64,
        "joint_position_limits_rad": [[-4.0, 4.0] for _ in range(7)],
        "max_abs_joint_velocity_rad_s": 8.0,
        "gripper_motor_position_limits_rad": [-0.2, 0.2],
        "ring_overflow_count": 0,
        "publisher_rejected_record_count": 0,
        "socket_send_failure_count": 0,
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
        "action_semantics": "exact_mitparam_consumed_at_control_tick",
        "action_interpolation": "forbidden",
        "envelope_count": 1,
        "control_tick_count": 3,
        "control_tick_first": 100,
        "control_tick_last": 102,
        "arm_tx_expected_frame_count": 21,
        "arm_tx_full_frame_write_count": 21,
        "arm_tx_failed_frame_write_count": 0,
        "gripper_tx_expected_frame_count": 3,
        "gripper_tx_full_frame_write_count": 3,
        "gripper_tx_failed_frame_write_count": 0,
        "socketcan_tx_enqueue_failure_tick_count": 0,
        "frame_count": 1,
        "hash_inventory": {},
    }
    summary.update(changes)
    return summary


def _exact_robot_record(tick: int, observation_ns: int) -> dict:
    q = [tick / 1000.0 + index * 0.01 for index in range(7)]
    return {
        "schema": "wam_openarm_exact_control_record_v3",
        "control_tick_seq": tick,
        "observation_monotonic_ns": observation_ns,
        "reference_generation": 17,
        "reference_source_monotonic_ns": observation_ns - 100_000,
        "reference_snapshot_valid": True,
        "control_call_begin_monotonic_ns": observation_ns + 50_000,
        "arm_call_return_monotonic_ns": observation_ns + 100_000,
        "gripper_call_begin_monotonic_ns": observation_ns + 110_000,
        "gripper_call_return_monotonic_ns": observation_ns + 160_000,
        "control_call_end_monotonic_ns": observation_ns + 170_000,
        "follower_q_rad": q,
        "follower_qd_rad_s": [0.5] * 7,
        "follower_gripper_q_rad": -0.07,
        "follower_gripper_qd_rad_s": 0.01,
        "follower_gripper_motor_q_rad": -0.075,
        "follower_gripper_motor_qd_rad_s": 0.015,
        "follower_arm_mit_kp": [240, 240, 240, 240, 40, 48, 40],
        "follower_arm_mit_kd": [3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3],
        "follower_arm_mit_q": q,
        "follower_arm_mit_dq": [0.4] * 7,
        "follower_arm_mit_tau": [0] * 7,
        "follower_gripper_mit_kp": 25.0,
        "follower_gripper_mit_kd": 0.3,
        "follower_gripper_mit_q": -0.08,
        "follower_gripper_mit_dq": 0.0,
        "follower_gripper_mit_tau": 0.0,
        "arm_tx_expected_frame_count": 7,
        "arm_tx_full_frame_write_count": 7,
        "arm_tx_failed_frame_write_count": 0,
        "gripper_tx_expected_frame_count": 1,
        "gripper_tx_full_frame_write_count": 1,
        "gripper_tx_failed_frame_write_count": 0,
        "socketcan_tx_enqueue_success": True,
    }




def _write_valid_robot_artifacts(tmp_path: Path, *, frame_index: int = 7) -> dict:
    from ausc.perception.robot.robot_session_capture import RobotSessionCapture
    from ausc.perception.robot.robot_state_stream import RobotStateContract

    joint_names = tuple(f"openarm_left_joint{index}" for index in range(1, 8))
    contract = RobotStateContract(
        side="left",
        controller_mode="unilateral",
        joint_names=joint_names,
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
    nominal_ns = int(round((100.0 + frame_index / 30.0) * 1_000_000_000))

    def envelope(sequence: int, first_tick: int, first_ns: int, count: int) -> bytes:
        records = [
            _exact_robot_record(first_tick + offset, first_ns + offset * 1_000_000)
            for offset in range(count)
        ]
        payload = {
            "schema": "wam_openarm_control_envelope_v3",
            "envelope_seq": sequence,
            "clock_id": "CLOCK_MONOTONIC",
            "publisher_monotonic_ns": (
                records[-1]["control_call_end_monotonic_ns"] + 500_000
            ),
            "controller_pid": 4242,
            "controller_start_token": "proc-start-991",
            "side": "left",
            "controller_mode": "unilateral",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "joint_names": list(joint_names),
            "urdf_sha256": contract.urdf_sha256,
            "joint_mapper_sha256": contract.joint_mapper_sha256,
            "pair_contract_sha256": contract.pair_contract_sha256,
            "controller_binary_sha256": contract.controller_binary_sha256,
            "controller_source_manifest_sha256": (
                contract.controller_source_manifest_sha256
            ),
            "follower_config_sha256": contract.follower_config_sha256,
            "openarm_sdk_sha256": contract.openarm_sdk_sha256,
            "publisher_nominal_hz": 100,
            "control_nominal_hz": 1000,
            "send_status_observable": True,
            "send_status_reason": "openarm_can_1_2_2_full_frame_write_results",
            "send_status_semantics": (
                "linux_socketcan_full_frame_write_tx_enqueue_not_motor_ack"
            ),
            "motor_ack_observable": False,
            "motor_ack_reason": (
                "socketcan_tx_enqueue_does_not_acknowledge_motor_receipt_or_execution"
            ),
            "ring_overflow_count": 0,
            "publisher_rejected_record_count": 0,
            "socket_send_failure_count": 0,
            "records": records,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    robot = RobotSessionCapture(
        tmp_path, contract, preflight_min_envelopes=2,
        preflight_duration_ns=10_000_000,
    )
    robot.ingest_datagram(envelope(48, 70, nominal_ns - 31_000_000, 10))
    robot.ingest_datagram(envelope(49, 80, nominal_ns - 21_000_000, 10))
    robot.start(now_monotonic_ns=nominal_ns - 11_000_000)
    robot.ingest_datagram(envelope(50, 90, nominal_ns - 10_000_000, 16))
    robot.record_frame_evidence(
        camera_id="orbbec-left-rgbd",
        frame_index=frame_index,
        exposure_lower_ns=nominal_ns - 500_000,
        exposure_nominal_ns=nominal_ns,
        exposure_upper_ns=nominal_ns + 500_000,
        now_monotonic_ns=nominal_ns + 6_000_000,
        camera_profile_sha256=contract.camera_profile_sha256,
        timing_artifact_sha256=contract.timing_artifact_sha256,
    )
    return dict(robot.finish(now_monotonic_ns=nominal_ns + 6_000_000))


def _write_valid_v4_robot_artifacts(
    tmp_path: Path, *, degenerate: bool = False,
) -> tuple[dict, object]:
    """Use the canonical robot producer to commit one exact v4 sidecar."""

    from ausc.perception.robot.tests.test_robot_session_capture import (
        _contract,
        _new_capture,
        _record_frame,
        _start_and_record_v4,
    )

    robot = _new_capture(tmp_path)
    _start_and_record_v4(robot, all_zero_q=degenerate)
    _record_frame(robot)
    return dict(robot.finish(now_monotonic_ns=1_030_000_000)), _contract()


def test_robot_summary_rejects_non_left_production_pair_before_binding(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()

    with pytest.raises(RuntimeError, match="left.*can1.*can3"):
        capture.set_robot_summary(_robot_summary(
            side="right",
            leader_can_interface="can0",
            follower_can_interface="can2",
        ))

    assert capture._robot_summary is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("controller_source_manifest_sha256", "", "source.*SHA-256"),
        ("follower_config_sha256", "x" * 64, "config.*SHA-256"),
        ("openarm_sdk_sha256", "A" * 64, "SDK.*SHA-256"),
        ("camera_profile_sha256", "bad", "camera profile.*SHA-256"),
        ("controller_pid", 0, "controller process identity"),
        ("ring_overflow_count", 1, "ring_overflow_count.*zero"),
        ("publisher_rejected_record_count", 1, "publisher_rejected.*zero"),
        ("socket_send_failure_count", 1, "socket_send_failure.*zero"),
        ("send_status_observable", False, "SocketCAN TX.*observable"),
        ("send_status_reason", "guessed", "send status reason"),
        ("send_status_semantics", "motor_acknowledged", "SocketCAN TX enqueue"),
        ("motor_ack_observable", True, "motor acknowledgement"),
        ("feedback_liveness_observable", True, "feedback liveness.*unobservable"),
        ("feedback_liveness_reason", "guessed", "feedback liveness reason"),
        ("training_eligible", True, "training_eligible=false"),
        (
            "eligibility_blockers",
            [],
            "feedback liveness blocker",
        ),
        ("arm_tx_full_frame_write_count", 20, "SocketCAN TX counts"),
        ("arm_tx_failed_frame_write_count", 1, "SocketCAN TX counts"),
        (
            "socketcan_tx_enqueue_failure_tick_count",
            1,
            "SocketCAN TX counts",
        ),
        ("action_semantics", "interpolated", "exact MIT"),
        ("action_interpolation", "linear", "interpolation.*forbidden"),
    ],
)
def test_robot_summary_fail_closed_contract_fields(tmp_path, field, value, message):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()

    with pytest.raises(RuntimeError, match=message):
        capture.set_robot_summary(_robot_summary(**{field: value}))

    assert capture._robot_summary is None


def test_abort_quarantines_final_robot_names_but_preserves_forensic_bytes(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    originals = {}
    for index, name in enumerate(
        (
            "robot_capture_summary.json",
            "robot_control_envelopes.jsonl",
            "robot_exact_control.csv",
            "robot_frame_index.jsonl",
        ),
        start=1,
    ):
        raw = f"forensic-{index}".encode()
        (tmp_path / name).write_bytes(raw)
        originals[name] = raw
    partial = tmp_path / "robot_control_envelopes.jsonl.partial"
    partial.write_bytes(b"partial-forensic")

    capture.abort("injected downstream failure")

    for name, raw in originals.items():
        assert not (tmp_path / name).exists()
        assert (tmp_path / f"{name}.aborted").read_bytes() == raw
    assert partial.read_bytes() == b"partial-forensic"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False
    assert manifest["robot_abort_quarantine"]["commit_marker_removed"] is True


def test_v3_robot_sidecar_is_replayed_and_bound_into_manifest_and_seal(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    robot_summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    assert robot_summary["feedback_liveness_observable"] is False
    assert robot_summary["training_eligible"] is False
    assert robot_summary["eligibility_blockers"] == [
        "robot_feedback_liveness_unobservable"
    ]
    capture.set_robot_summary(robot_summary)

    result = capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    expected = {
        "robot_control_envelopes.jsonl", "robot_exact_control.csv",
        "robot_frame_index.jsonl", "robot_capture_summary.json",
    }
    assert result["robot"] == robot_summary
    assert manifest["robot"] == robot_summary
    assert seal["robot"] == robot_summary
    assert expected.issubset(manifest["artifacts"])
    assert expected.issubset(seal["artifacts"])
    assert manifest["training_eligible"] is False


def test_v4_robot_sidecar_requires_and_replays_actual_frozen_contract(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, contract = _write_valid_v4_robot_artifacts(tmp_path)

    capture.set_robot_summary(summary, contract=contract)
    result = capture.finalize()

    assert summary["schema"] == "wam_openarm_exact_control_capture_summary_v4"
    assert summary["feedback_liveness_observable"] is True
    assert summary["training_eligible"] is True
    assert summary["eligibility_blockers"] == []
    assert result["robot"] == summary
    assert json.loads((tmp_path / "manifest.json").read_text())["robot"] == summary


def test_v4_robot_sidecar_rejects_missing_actual_contract(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, _contract = _write_valid_v4_robot_artifacts(tmp_path)

    with pytest.raises(RuntimeError, match="v4.*actual frozen RobotStateContract"):
        capture.set_robot_summary(summary)

    assert capture._robot_summary is None


def test_v4_robot_sidecar_rejects_contract_not_matching_raw_activation(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, contract = _write_valid_v4_robot_artifacts(tmp_path)
    forged = replace(contract, controller_binary_sha256="9" * 64)

    with pytest.raises(RuntimeError, match="canonical replay failed"):
        capture.set_robot_summary(summary, contract=forged)

    assert capture._robot_summary is None


def test_v4_robot_sidecar_rejects_canonical_degenerate_trajectory(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, contract = _write_valid_v4_robot_artifacts(
        tmp_path, degenerate=True,
    )
    assert summary["training_eligible"] is False
    assert "robot_observation_trajectory_degenerate" in summary[
        "eligibility_blockers"
    ]

    with pytest.raises(RuntimeError, match="v4.*training eligible"):
        capture.set_robot_summary(summary, contract=contract)

    assert capture._robot_summary is None


def test_v4_robot_raw_degeneracy_cannot_be_overridden_by_forged_pass(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, contract = _write_valid_v4_robot_artifacts(
        tmp_path, degenerate=True,
    )
    summary.update({
        "trajectory_nondegenerate": True,
        "command_response_consistent": True,
        "command_response_violation_joint_names": [],
        "training_eligible": True,
        "eligibility_blockers": [],
    })
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="canonical replay failed"):
        capture.set_robot_summary(summary, contract=contract)

    assert capture._robot_summary is None


def test_v4_robot_replay_uses_one_pinned_nofollow_descriptor_set(
    tmp_path, monkeypatch,
):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary, contract = _write_valid_v4_robot_artifacts(tmp_path)

    from ausc.perception.robot import robot_session_capture as robot_capture_module

    original = robot_capture_module.validate_robot_capture_artifacts
    observed = {}

    def require_pinned(session_dir, activated_contract, *, artifact_fds=None):
        observed["fds"] = artifact_fds
        assert set(artifact_fds) == set(
            perception_capture._ROBOT_FINAL_ARTIFACT_NAMES
        )
        assert all(stat.S_ISREG(os.fstat(fd).st_mode) for fd in artifact_fds.values())
        return original(
            session_dir, activated_contract, artifact_fds=artifact_fds,
        )

    monkeypatch.setattr(
        robot_capture_module, "validate_robot_capture_artifacts", require_pinned,
    )
    capture.set_robot_summary(summary, contract=contract)

    assert observed["fds"] is not None


def test_production_robot_binding_rejects_v2_commissioning_summary(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()

    with pytest.raises(RuntimeError, match="requires v3"):
        capture.set_robot_summary(_robot_summary(
            schema="wam_openarm_exact_control_capture_summary_v2",
            status="exact_control_capture_validated_send_status_unobservable",
            send_status_observable=False,
            send_status_reason="upstream_mit_control_all_returns_void",
            training_eligible=False,
            eligibility_blockers=["robot_action_send_status_unobservable"],
        ))

    assert capture._robot_summary is None


def test_robot_semantic_replay_runs_once_and_finalize_reuses_bound_receipt(
    tmp_path, monkeypatch,
):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)

    from ausc.perception.robot import robot_session_capture as robot_capture_module

    replay_calls = []
    original_replay = robot_capture_module.validate_robot_capture_artifacts

    def counted_replay(session_dir, contract):  # noqa: ANN001
        replay_calls.append((Path(session_dir), contract))
        return original_replay(session_dir, contract)

    monkeypatch.setattr(
        robot_capture_module, "validate_robot_capture_artifacts", counted_replay,
    )
    capture.set_robot_summary(summary)
    assert len(replay_calls) == 1
    replay_contract = replay_calls[0][1]
    assert replay_contract.joint_position_limits_rad == tuple(
        (-4.0, 4.0) for _ in range(7)
    )
    assert replay_contract.max_abs_joint_velocity_rad_s == 8.0
    assert replay_contract.gripper_motor_position_limits_rad == (-0.2, 0.2)

    robot_hash_reads = []
    original_sha256 = perception_capture._sha256

    def counted_sha256(path):  # noqa: ANN001
        path = Path(path)
        if path.name in perception_capture._ROBOT_FINAL_ARTIFACT_NAMES:
            robot_hash_reads.append(path.name)
        return original_sha256(path)

    monkeypatch.setattr(perception_capture, "_sha256", counted_sha256)
    capture.finalize()

    assert len(replay_calls) == 1
    assert robot_hash_reads == []


def test_robot_setter_rejects_data_artifact_hash_mismatch(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    with (tmp_path / "robot_control_envelopes.jsonl").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(RuntimeError, match="canonical replay failed"):
        capture.set_robot_summary(summary)

    assert capture._robot_summary is None


def test_finalize_revalidates_robot_sidecar_and_quarantines_post_bind_tamper(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    capture.set_robot_summary(summary)
    with (tmp_path / "robot_frame_index.jsonl").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(RuntimeError, match="artifact identity changed"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert not (tmp_path / "robot_capture_summary.json").exists()
    assert (tmp_path / "robot_capture_summary.json.aborted").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False


def _rewrite_robot_summary(tmp_path: Path, summary: dict) -> None:
    (tmp_path / "robot_capture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _refresh_robot_inventory(tmp_path: Path, summary: dict, name: str) -> None:
    data = (tmp_path / name).read_bytes()
    summary["hash_inventory"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def test_robot_replay_rejects_summary_envelope_count_not_in_raw_stream(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    summary["envelope_count"] = 2
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="summary replay mismatch"):
        capture.set_robot_summary(summary)


def test_robot_replay_rejects_exact_control_csv_tick_duplication(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    path = tmp_path / "robot_exact_control.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[1]["control_tick_seq"] = rows[0]["control_tick_seq"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_robot_inventory(tmp_path, summary, "robot_exact_control.csv")
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="exact control CSV.*tick"):
        capture.set_robot_summary(summary)


def test_robot_replay_rejects_frame_index_not_closing_on_camera_capture(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    path = tmp_path / "robot_frame_index.jsonl"
    row = json.loads(path.read_text())
    row["frame_index"] = 8
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _refresh_robot_inventory(tmp_path, summary, "robot_frame_index.jsonl")
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="frame index.*camera capture"):
        capture.set_robot_summary(summary)


def test_robot_replay_rejects_frame_exact_action_hash_mismatch(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    path = tmp_path / "robot_frame_index.jsonl"
    row = json.loads(path.read_text())
    row["exact_action_sha256"] = "0" * 64
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _refresh_robot_inventory(tmp_path, summary, "robot_frame_index.jsonl")
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="exact action range hash"):
        capture.set_robot_summary(summary)


def test_robot_replay_rejects_duplicate_json_fields(tmp_path):
    capture, _, _ = _capture(tmp_path, require_robot=True)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_robot_artifacts(tmp_path, frame_index=7)
    path = tmp_path / "robot_control_envelopes.jsonl"
    raw = path.read_text()
    assert '"envelope_seq":50' in raw
    path.write_text(raw.replace(
        '"envelope_seq":50', '"envelope_seq":49,"envelope_seq":50', 1,
    ))
    _refresh_robot_inventory(tmp_path, summary, "robot_control_envelopes.jsonl")
    _rewrite_robot_summary(tmp_path, summary)

    with pytest.raises(RuntimeError, match="duplicate JSON field"):
        capture.set_robot_summary(summary)


def test_capture_requires_device_model_and_runtime_provenance(tmp_path):
    with pytest.raises(ValueError, match="provenance.*model"):
        DiagnosticPerceptionCapture(
            tmp_path, fps=30, provenance={"camera": {}, "runtime": {}},
        )


def test_disk_preflight_fails_before_any_manifest_or_writer_is_created(tmp_path):
    capture, writers, _ = _capture(tmp_path, free_bytes=100)

    with pytest.raises(RuntimeError, match="free disk"):
        capture.start()

    assert writers == []
    assert not (tmp_path / "manifest.json").exists()


def test_pinned_ffmpeg_concat_copy_uses_fixed_argv_and_records_provenance(
    tmp_path, monkeypatch,
):
    executable = tmp_path / "pinned-ffmpeg"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    segment_dir = tmp_path / "rgb_segments"
    segment_dir.mkdir()
    segments = []
    for index in range(2):
        segment = segment_dir / f"segment_{index:06d}.mp4"
        segment.write_bytes(f"segment-{index}".encode())
        segments.append(segment)

    def probe(path: Path) -> dict:
        counts = {
            "segment_000000.mp4": 2,
            "segment_000001.mp4": 3,
            "rgb.finalizing.mp4": 5,
        }
        return {
            "fps": 30.0,
            "width": 12,
            "height": 8,
            "frame_count": counts[Path(path).name],
        }

    observed = {"calls": []}

    def fake_run(argv, **kwargs):  # noqa: ANN001
        observed["calls"].append((list(argv), dict(kwargs)))
        cwd = Path(kwargs["cwd"])
        if "concat" in argv:
            observed["argv"] = list(argv)
            observed["kwargs"] = dict(kwargs)
            concat_name = argv[argv.index("-i") + 1]
            observed["concat"] = (cwd / concat_name).read_text()
            (cwd / argv[-1]).write_bytes(b"concat-copy-output")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=b"frame=5\nprogress=end\n",
            stderr=b"",
        )

    monkeypatch.setattr(perception_capture, "_default_video_mode_probe", probe)
    monkeypatch.setattr(perception_capture.subprocess, "run", fake_run)
    stitcher = perception_capture._PinnedFfmpegConcatStitcher(executable, digest)

    provenance = stitcher.prepare()
    result = stitcher(
        segments, tmp_path / "rgb.finalizing.mp4", 30.0, (12, 8),
    )

    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["cwd"] == str(tmp_path)
    assert observed["argv"][0] == str(executable.resolve())
    assert observed["argv"][observed["argv"].index("-c") + 1] == "copy"
    assert observed["concat"] == (
        "file 'rgb_segments/segment_000000.mp4'\n"
        "file 'rgb_segments/segment_000001.mp4'\n"
    )
    assert provenance["sha256"] == digest
    assert provenance["shell"] is False
    assert result["frame_count"] == 5
    assert result["provenance"] == provenance
    assert result["packet_validation"] == {
        "verified": True,
        "decoded_frame_count": 5,
        "expected_frame_count": 5,
        "method": "ffmpeg_decode_null_muxer",
    }
    assert len(observed["calls"]) == 2
    packet_argv = observed["calls"][1][0]
    assert "-c:v" not in packet_argv
    assert packet_argv[packet_argv.index("-progress") + 1] == "pipe:1"
    assert not list(tmp_path.glob(".rgb.concat.*.txt"))


def test_wrong_pinned_ffmpeg_digest_fails_start_before_session_artifacts(tmp_path):
    executable = tmp_path / "pinned-ffmpeg"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    capture = DiagnosticPerceptionCapture(
        tmp_path,
        fps=30.0,
        provenance=PROVENANCE,
        ffmpeg_path=executable,
        ffmpeg_sha256="0" * 64,
        min_free_bytes=1,
    )

    with pytest.raises(RuntimeError, match="ffmpeg concat-copy SHA-256 mismatch"):
        capture.start()

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "rgb_segments").exists()
    assert not (tmp_path / "depth").exists()


def test_video_writer_must_open_on_the_first_camera_frame(tmp_path):
    capture, writers, _ = _capture(tmp_path, opened=False)
    capture.start()

    with pytest.raises(RuntimeError, match="video writer.*open"):
        capture.write_frame(_frame(7), _tracking(_frame(7)))

    assert writers[0].released is True
    assert json.loads((tmp_path / "manifest.json").read_text())["training_eligible"] is False


def test_finalize_reopens_video_and_seals_exact_diagnostic_artifacts(tmp_path):
    capture, writers, validated = _capture(tmp_path)
    capture.start()
    capture.record_event({"event": "attempt_started", "attempt": 1})
    for index in (7, 8, 9):
        frame = _frame(index, value=index)
        capture.write_frame(frame, _tracking(frame))

    result = capture.finalize()

    assert result["status"] == "diagnostic_complete"
    assert result["training_eligible"] is False
    assert writers[0].released is True
    assert validated[-1] == tmp_path / "rgb.mp4"
    assert validated[:-1] == [
        tmp_path / "rgb_segments" / "segment_000000.mp4",
    ]
    expected = {
        "rgb.mp4", "rgb_times.csv", "tracker.jsonl", "events.jsonl",
        "depth/depth_times.csv", "depth/depth_000007.npy.zlib", "depth/depth_000008.npy.zlib",
        "depth/depth_000009.npy.zlib",
        "capture_checkpoint.json", "manifest.json", "DIAGNOSTIC_FINALIZATION.json",
    }
    assert expected.issubset(
        {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()}
    )
    assert not (tmp_path / "depth_times.csv").exists()
    assert not (tmp_path / "FINALIZATION.json").exists()

    with (tmp_path / "rgb_times.csv").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [int(row["frame_index"]) for row in rows] == [7, 8, 9]
    tracker = [json.loads(line) for line in (tmp_path / "tracker.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in tracker] == [7, 8, 9]
    assert tracker[0]["points"][0]["name"] == "R1"

    from ausc.perception import depth_codec
    depth_indexes: list[int] = []
    for path in sorted((tmp_path / "depth").glob("*.npy.zlib")):
        depth = depth_codec.load_depth(path)
        assert depth.shape == (8, 12)
        depth_indexes.append(
            int(path.name.removeprefix("depth_").removesuffix(".npy.zlib"))
        )
    assert depth_indexes == [7, 8, 9]

    manifest_raw = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    assert manifest["capture_mode"] == "diagnostic_perception"
    assert manifest["training_eligible"] is False
    assert "promotion_contract_version" not in manifest
    assert manifest["provenance"] == PROVENANCE
    assert manifest["n_rgb_frames"] == manifest["n_depth_frames"] == 3
    assert manifest["timeline"] == {
        "camera_index_coverage_complete": True,
        "duration_s": pytest.approx(2 / 30),
        "first_host_t": pytest.approx(100 + 7 / 30),
        "last_host_t": pytest.approx(100 + 9 / 30),
        "max_host_gap_s": pytest.approx(1 / 30),
        "missing_camera_indexes": 0,
        "observed_fps": pytest.approx(30.0),
        "model_gap_evidence": {
            "same_frame_count": 3,
            "carried_forward_count": 0,
            "unavailable_count": 0,
            "max_tracking_age_s": 0.0,
        },
    }
    assert seal["schema_version"] == "wam_diagnostic_finalization_v1"
    assert seal["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert seal["artifacts"]["rgb.mp4"]["sha256"] == hashlib.sha256(
        (tmp_path / "rgb.mp4").read_bytes()
    ).hexdigest()
    assert "training_eligible" not in seal


def test_audio_summary_binds_all_final_audio_files_into_diagnostic_seal(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    summary = _write_valid_audio_artifacts(tmp_path)
    capture.set_audio_summary(summary)

    result = capture.finalize()

    assert result["audio"]["sha256"] == summary["sha256"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    expected = {
        "audio.wav", "audio_capture.json", "audio_input_times.jsonl",
        "audio_times.jsonl",
    }
    assert expected.issubset(manifest["artifacts"])
    assert expected.issubset(seal["artifacts"])
    assert manifest["audio"]["n_samples"] == 1_600


def test_audio_validator_accepts_artifacts_from_real_44100_to_16000_producer(tmp_path):
    from auscultation.ausc.audio.audio_stream import AudioStream

    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(7)
    capture.write_frame(frame, _tracking(frame))
    audio = AudioStream(
        0, sr_in=44_100, sr_model=16_000, channels=1,
        buffer_seconds=0.25, block_ms=100.0,
    )
    audio.begin_recording(tmp_path, t0=10.0)
    rng = np.random.default_rng(22)
    quiet_pcm = rng.integers(-8, 9, size=4_410, dtype=np.int16)
    quiet = quiet_pcm.astype(np.float32) / 32_767.0
    audio._ingest(quiet[:2_205], 10.05)
    audio._ingest(quiet[2_205:], 10.1)
    produced = audio.end_recording(t_end=10.1)

    capture.set_audio_summary({
        "path": "audio.wav",
        "sample_rate": produced.sr,
        "input_sample_rate": 44_100,
        "n_samples": produced.n_samples,
        "t_start": produced.t_start,
        "t_end": produced.t_end,
        "status": produced.status,
        "peak_db": produced.peak_db,
        "sha256": produced.sha256,
    })

    result = capture.finalize()

    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    authenticity = metadata["signal_authenticity"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    assert result["audio"]["n_samples"] == produced.n_samples
    assert result["audio_authenticity"] == authenticity
    assert manifest["audio_authenticity"] == authenticity
    assert seal["audio_authenticity"] == authenticity


def test_audio_summary_rejects_undecodable_wav(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    (tmp_path / "audio.wav").write_bytes(b"RIFF-not-a-decodable-wave")
    _bind_current_audio_sha(tmp_path, summary)

    with pytest.raises(RuntimeError, match="diagnostic audio WAV.*decode"):
        capture.set_audio_summary(summary)


@pytest.mark.parametrize(
    ("channels", "sample_width", "wav_rate", "frame_count", "expected"),
    [
        (2, 2, 16_000, 1_600, "mono"),
        (1, 1, 16_000, 1_600, "PCM16"),
        (1, 2, 8_000, 1_600, "sample rate"),
        (1, 2, 16_000, 1_599, "frame count"),
    ],
)
def test_audio_summary_rejects_wav_mode_or_frame_count_mismatch(
    tmp_path, channels, sample_width, wav_rate, frame_count, expected,
):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    frames = (
        np.full(frame_count * channels, 4_096, dtype="<i2").tobytes()
        if sample_width == 2 else bytes([255]) * frame_count * channels
    )
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(wav_rate)
        wav.writeframes(frames)
    _bind_current_audio_sha(tmp_path, summary)

    with pytest.raises(RuntimeError, match=expected):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_coherent_non_16000_model_rate(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    pcm = np.full(800, 4_096, dtype="<i2")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(pcm.tobytes())
    summary.update(sample_rate=8_000, n_samples=800)
    metadata_path = tmp_path / "audio_capture.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(sample_rate=8_000, n_samples=800)
    metadata_path.write_text(json.dumps(metadata))
    timeline_path = tmp_path / "audio_times.jsonl"
    timeline = json.loads(timeline_path.read_text())
    timeline.update(sample_rate=8_000, sample_count=800)
    timeline_path.write_text(json.dumps(timeline) + "\n")
    _bind_current_audio_sha(tmp_path, summary)

    with pytest.raises(RuntimeError, match="model sample rate must be 16000"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_silent_wav_even_if_summary_reports_silence(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(1_600, dtype="<i2").tobytes())
    summary["peak_db"] = -180.0
    _bind_current_audio_sha(tmp_path, summary)

    with pytest.raises(RuntimeError, match="diagnostic audio WAV is silent"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_peak_claim_that_disagrees_with_pcm(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    summary["peak_db"] = -3.0

    with pytest.raises(RuntimeError, match="diagnostic audio peak.*WAV"):
        capture.set_audio_summary(summary)


def test_audio_summary_recomputes_authenticity_instead_of_trusting_metadata(
    tmp_path,
):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    metadata_path = tmp_path / "audio_capture.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["signal_authenticity"]["status"] = "FAIL"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="diagnostic audio metadata"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_pcm_that_disagrees_with_authenticity_receipt(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    pcm = np.zeros(1_600, dtype="<i2")
    pcm[0] = 4_096  # Correct advertised peak, but only one non-zero sample.
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    _bind_current_audio_sha(tmp_path, summary)

    with pytest.raises(RuntimeError, match="diagnostic audio metadata"):
        capture.set_audio_summary(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_offset", 1),
        ("sample_count", 1_599),
        ("sample_rate", 8_000),
        ("input_sample_rate", 48_000),
        ("t_start", 10.01),
        ("t_end", 10.2),
    ],
)
def test_audio_summary_rejects_invalid_model_timeline(tmp_path, field, value):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_times.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="diagnostic audio timeline"):
        capture.set_audio_summary(summary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_sample_offset", 1),
        ("input_sample_count", 4_409),
        ("t_start", 10.01),
        ("t_end", 10.2),
    ],
)
def test_audio_summary_rejects_invalid_input_timeline(tmp_path, field, value):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_input_times.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="diagnostic audio input timeline"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_missing_input_source_clock_timestamp(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_input_times.jsonl"
    row = json.loads(path.read_text())
    del row["source_block_start_t"]
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="input timeline.*incomplete"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_input_source_clock_gap(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    rows = [
        {
            "input_sample_offset": 0,
            "input_sample_count": 2_205,
            "t_start": 10.0,
            "t_end": 10.05,
            "source_block_start_t": 10.0,
            "source_block_end_t": 10.05,
        },
        {
            "input_sample_offset": 2_205,
            "input_sample_count": 2_205,
            "t_start": 10.05,
            "t_end": 10.1,
            "source_block_start_t": 19.0,
            "source_block_end_t": 19.05,
        },
    ]
    (tmp_path / "audio_input_times.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    with pytest.raises(RuntimeError, match="input source clock discontinuity"):
        capture.set_audio_summary(summary)


def test_audio_summary_allows_final_input_source_block_end_after_end_clip(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_input_times.jsonl"
    row = json.loads(path.read_text())
    row["source_block_end_t"] = 10.2
    path.write_text(json.dumps(row) + "\n")

    capture.set_audio_summary(summary)

    assert capture._audio_summary["n_samples"] == 1_600


def test_audio_summary_rejects_audio_start_outside_50ms_t0_coverage(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_capture.json"
    metadata = json.loads(path.read_text())
    metadata["t0_monotonic"] = 9.0
    path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="audio start.*50 ms"):
        capture.set_audio_summary(summary)


def test_audio_summary_rejects_input_timeline_metadata_count_mismatch(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    path = tmp_path / "audio_capture.json"
    metadata = json.loads(path.read_text())
    metadata["partial_n_samples"] = 4_409
    path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="diagnostic audio metadata"):
        capture.set_audio_summary(summary)


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        ("summary", "sample_rate", 16_000.9),
        ("summary", "n_samples", "1600"),
        ("metadata", "partial_n_samples", 4_410.9),
        ("metadata", "input_sample_rate", "44100"),
        ("model_timeline", "sample_offset", 0.9),
        ("model_timeline", "sample_count", "1600"),
        ("input_timeline", "input_sample_offset", 0.9),
        ("input_timeline", "input_sample_count", "4410"),
    ],
)
def test_audio_summary_rejects_coerced_non_integer_contract_fields(
    tmp_path, artifact, field, value,
):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    summary = _write_valid_audio_artifacts(tmp_path)
    if artifact == "summary":
        summary[field] = value
    else:
        paths = {
            "metadata": tmp_path / "audio_capture.json",
            "model_timeline": tmp_path / "audio_times.jsonl",
            "input_timeline": tmp_path / "audio_input_times.jsonl",
        }
        path = paths[artifact]
        payload = json.loads(path.read_text())
        payload[field] = value
        path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(RuntimeError, match="exact integer"):
        capture.set_audio_summary(summary)


def test_finalize_rejects_video_frame_count_mismatch_without_a_seal(tmp_path):
    capture, _, _ = _capture(tmp_path, validator=lambda _: 0)
    capture.start()
    frame = _frame(1)
    capture.write_frame(frame, _tracking(frame))

    with pytest.raises(RuntimeError, match="decoded 0 frames.*recorded 1"):
        capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert not (tmp_path / "FINALIZATION.json").exists()


def test_capture_rejects_fps_that_disagrees_with_negotiated_camera_mode(tmp_path):
    provenance = json.loads(json.dumps(PROVENANCE))
    provenance["camera"]["negotiated_mode"] = {
        "width": 12, "height": 8, "fps": 30.0,
    }

    with pytest.raises(ValueError, match="fps.*negotiated camera FPS"):
        DiagnosticPerceptionCapture(tmp_path, fps=20.0, provenance=provenance)


def test_finalize_rejects_any_camera_index_gap_instead_of_sealing(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    for index in (1, 3):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    with pytest.raises(RuntimeError, match="camera frame index coverage.*missing 1"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_tracker_rows_preserve_source_frame_association_and_exact_inference_timeline(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    for index in (10, 11):
        frame = _frame(index)
        tracking = _tracking(frame)
        tracking.update({
            "source_tracking_frame_index": 9,
            "source_tracking_host_t_s": frame.t - (index - 9) / 30.0,
            "source_tracking_device_t_ms": frame.device_t_ms - (index - 9) * 33.0,
            "tracking_association": "carried_forward",
            "tracking_age_s": (index - 9) / 30.0,
        })
        capture.write_frame(frame, tracking)
    capture.finalize()

    rows = [json.loads(line) for line in (tmp_path / "tracker.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [10, 11]
    assert [row["source_tracking_frame_index"] for row in rows] == [9, 9]
    assert [row["tracking_association"] for row in rows] == [
        "carried_forward", "carried_forward",
    ]
    inference = [
        json.loads(line)
        for line in (tmp_path / "inference.jsonl").read_text().splitlines()
    ]
    assert len(inference) == 1
    assert inference[0]["source_tracking_frame_index"] == 9
    assert inference[0]["first_associated_camera_frame_index"] == 10


def test_capture_rejects_future_or_implicitly_same_frame_tracking(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(11)
    tracking = _tracking(frame)
    tracking.update({
        "source_tracking_frame_index": 12,
        "tracking_association": "carried_forward",
        "tracking_age_s": 0.0,
    })

    with pytest.raises(RuntimeError, match="future tracking frame"):
        capture.write_frame(frame, tracking)

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_start_and_end_boundaries_are_sealed_with_offline_recompute_contract(tmp_path):
    start_boundary = {
        "start_host_t_s": 99.0,
        "camera_start_boundary_index": 4,
        "model_start_boundary_index": 3,
    }
    capture, _, _ = _capture(tmp_path, recording_boundary=start_boundary)
    capture.start()
    frame = _frame(5)
    capture.write_frame(frame, _tracking(frame))
    end_boundary = {
        **start_boundary,
        "last_submitted_camera_index": 5,
        "last_completed_camera_index": 5,
        "last_completed_model_index": 4,
        "end_camera_frame_has_completed_model_result": False,
        "inference_timeline_policy": "latest_only_recomputable_from_pinned_rgbd",
        "offline_model_recompute_from_pinned_rgbd": True,
    }
    capture.set_finalization_context(end_boundary)

    result = capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    assert manifest["recording_boundary"] == start_boundary
    assert manifest["timeline"]["recording_boundary"] == start_boundary
    assert manifest["timeline"]["end_boundary"] == end_boundary
    assert manifest["capture_boundary"] == end_boundary
    assert result["capture_boundary"] == end_boundary
    assert seal["capture_boundary"] == end_boundary


def test_finalize_rejects_end_boundary_that_does_not_close_on_last_written_frame(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    for index in (5, 6):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))
    capture.set_finalization_context({
        "camera_start_boundary_index": 4,
        "last_submitted_camera_index": 6,
        "last_completed_camera_index": 5,
    })

    with pytest.raises(RuntimeError, match="capture boundary.*last written frame"):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_duplicate_camera_frame_is_not_recorded_twice(tmp_path):
    capture, writers, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(11)

    assert capture.write_frame(frame, _tracking(frame)) is True
    assert capture.write_frame(frame, _tracking(frame)) is False
    capture.finalize()

    assert len(writers[0].frames) == 1


def test_checkpoint_flushes_indexes_and_states_mp4_recovery_limit(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(11)

    capture.write_frame(frame, _tracking(frame))

    checkpoint = json.loads((tmp_path / "capture_checkpoint.json").read_text())
    assert checkpoint["indexes_and_depth_durable"] is True
    assert checkpoint["video_container_recoverable"] is False
    assert checkpoint["sealed_video_frames"] == 0
    assert checkpoint["at_risk_video_frames"] == 1
    # The default bound is five seconds, independent of the configured FPS.
    assert checkpoint["video_segment_frames_limit"] == 150
    assert checkpoint["n_rgb_frames"] == checkpoint["n_depth_frames"] == 1
    with (tmp_path / "rgb_times.csv.inprogress").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [int(row["frame_index"]) for row in rows] == [11]
    assert (tmp_path / "depth" / "depth_000011.npy.zlib").is_file()
    capture.abort("test cleanup")


def test_video_is_durably_recoverable_in_bounded_segments_during_recording(tmp_path):
    capture, writers, validated = _capture(tmp_path, video_segment_frames=2)
    capture.start()

    for index in (20, 21, 22):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    sealed = tmp_path / "rgb_segments" / "segment_000000.mp4"
    active = tmp_path / "rgb_segments" / "segment_000001.inprogress.mp4"
    assert writers[0].released is True
    assert sealed.is_file()
    assert active.is_file()
    assert all(name != threading.current_thread().name for name in capture._test_validation_threads)
    journal = [
        json.loads(line)
        for line in (tmp_path / "video_segments.jsonl").read_text().splitlines()
    ]
    assert journal == [{
        "bytes": sealed.stat().st_size,
        "first_frame_index": 20,
        "fps": 30.0,
        "frame_count": 2,
        "height": 8,
        "last_frame_index": 21,
        "path": "rgb_segments/segment_000000.mp4",
        "segment_index": 0,
        "sha256": hashlib.sha256(sealed.read_bytes()).hexdigest(),
        "validation": "pending_async",
        "width": 12,
    }]
    checkpoint = json.loads((tmp_path / "capture_checkpoint.json").read_text())
    assert checkpoint["video_container_recoverable"] is True
    assert checkpoint["sealed_video_frames"] == 2
    assert checkpoint["at_risk_video_frames"] == 1
    assert checkpoint["video_segment_frames_limit"] == 2
    assert checkpoint["video_segments_durable"] is True
    assert checkpoint["sealed_video_segments_validated"] is False
    capture.abort("simulated operator abort")


def test_segment_rollover_does_not_decode_video_on_the_capture_ingress_thread(tmp_path):
    capture, writers, validated = _capture(tmp_path, video_segment_frames=2)
    capture.start()

    for index in (25, 26):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    assert writers[0].released is True
    assert (tmp_path / "rgb_segments" / "segment_000000.mp4").is_file()
    assert all(name != threading.current_thread().name for name in capture._test_validation_threads)
    capture.abort("test cleanup")


def test_finalize_stitches_all_segments_and_inventory_comes_from_durable_files(tmp_path):
    capture, writers, _ = _capture(tmp_path, video_segment_frames=2)
    capture.start()
    for index in range(30, 35):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    # A long capture must not depend on an ever-growing in-memory artifact list.
    if hasattr(capture, "_owned_artifacts"):
        capture._owned_artifacts.clear()
    result = capture.finalize()

    assert result["n_rgb_frames"] == 5
    assert len(writers) == 3
    assert all(writer.released for writer in writers)
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    assert seal["artifacts"]["rgb.mp4"]["bytes"] > 0
    assert sorted(
        name for name in seal["artifacts"] if name.startswith("rgb_segments/")
    ) == [
        "rgb_segments/segment_000000.mp4",
        "rgb_segments/segment_000001.mp4",
        "rgb_segments/segment_000002.mp4",
    ]
    assert sorted(
        name for name in seal["artifacts"] if name.startswith("depth/")
    ) == [
        "depth/depth_000030.npy.zlib",
        "depth/depth_000031.npy.zlib",
        "depth/depth_000032.npy.zlib",
        "depth/depth_000033.npy.zlib",
        "depth/depth_000034.npy.zlib",
        "depth/depth_times.csv",
    ]
    assert seal["depth_integrity"]["count"] == 5
    assert seal["depth_integrity"]["first_frame_index"] == 30
    assert seal["depth_integrity"]["last_frame_index"] == 34
    assert seal["artifacts"]["depth_integrity.jsonl"]["sha256"] == (
        seal["depth_integrity"]["index_sha256"]
    )
    # The platform keeps this result in memory for every /state request.  Its
    # payload must stay compact even though the complete on-disk seal inventories
    # every segment.
    assert "diagnostic_finalization" not in result
    assert result["diagnostic_seal"]["artifact_count"] == len(seal["artifacts"])
    assert "artifacts" not in result["diagnostic_seal"]


def test_model_gap_evidence_is_compactly_summarized_in_timeline(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    first = _frame(40)
    capture.write_frame(first, _tracking(first))
    second = _frame(41)
    capture.write_frame(second, {
        "frame_index": 41,
        "source_tracking_frame_index": 40,
        "source_tracking_host_t_s": first.t,
        "source_tracking_device_t_ms": first.device_t_ms,
        "tracking_association": "carried_forward",
        "tracking_age_s": second.t - first.t,
        "back_present": False,
        "initialized": False,
        "reason": "MODEL_STALE",
        "points": [],
    })
    third = _frame(42)
    capture.write_frame(third, {
        "frame_index": 42,
        "source_tracking_frame_index": None,
        "source_tracking_host_t_s": None,
        "source_tracking_device_t_ms": None,
        "tracking_association": "unavailable",
        "tracking_age_s": None,
        "back_present": False,
        "initialized": False,
        "reason": "MODEL_UNAVAILABLE_FOR_CAMERA_FRAME",
        "points": [],
    })

    timeline = capture.finalize()["timeline"]

    assert timeline["model_gap_evidence"] == {
        "same_frame_count": 1,
        "carried_forward_count": 1,
        "unavailable_count": 1,
        "max_tracking_age_s": pytest.approx(1 / 30),
    }


def test_orphan_recovery_stitches_only_sealed_segments_and_preserves_partial(tmp_path):
    segment_dir = tmp_path / "rgb_segments"
    segment_dir.mkdir()
    sealed = segment_dir / "segment_000000.mp4"
    partial = segment_dir / "segment_000001.inprogress.mp4"
    sealed.write_bytes(b"SEALED")
    partial.write_bytes(b"PARTIAL-DO-NOT-DELETE")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "recording",
        "requested_fps": 30.0,
        "training_eligible": False,
    }))
    (tmp_path / "video_segments.jsonl").write_text(json.dumps({
        "path": "rgb_segments/segment_000000.mp4",
        "segment_index": 0,
        "frame_count": 2,
        "first_frame_index": 40,
        "last_frame_index": 41,
        "fps": 30.0,
        "width": 12,
        "height": 8,
        "bytes": len(b"SEALED"),
        "sha256": hashlib.sha256(b"SEALED").hexdigest(),
    }) + "\n")

    def validate(path: Path) -> int:
        return 2

    def stitch(paths, output: Path, fps: float, size: tuple[int, int]) -> int:
        assert list(paths) == [sealed]
        assert fps == 30.0
        assert size == (12, 8)
        output.write_bytes(b"RECOVERED")
        return 2

    result = perception_capture.recover_orphan_diagnostic_video(
        tmp_path, video_validator=validate, video_stitcher=stitch,
    )

    assert result["status"] == "recovered_video"
    assert result["recovered_frame_count"] == 2
    assert result["source_segments"] == ["rgb_segments/segment_000000.mp4"]
    assert (tmp_path / "rgb.recovered.mp4").read_bytes() == b"RECOVERED"
    assert partial.read_bytes() == b"PARTIAL-DO-NOT-DELETE"
    assert not (tmp_path / "rgb.mp4").exists()
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_orphan_recovery_promotes_complete_recovering_file_idempotently(tmp_path):
    segment_dir = tmp_path / "rgb_segments"
    segment_dir.mkdir()
    sealed = segment_dir / "segment_000000.mp4"
    sealed.write_bytes(b"SEALED")
    (tmp_path / "video_segments.jsonl").write_text(json.dumps({
        "path": "rgb_segments/segment_000000.mp4",
        "segment_index": 0,
        "frame_count": 2,
        "first_frame_index": 40,
        "last_frame_index": 41,
        "fps": 30.0,
        "width": 12,
        "height": 8,
        "bytes": len(b"SEALED"),
        "sha256": hashlib.sha256(b"SEALED").hexdigest(),
    }) + "\n")
    recovering = tmp_path / "rgb.recovering.mp4"
    recovering.write_bytes(b"ALREADY-COMPLETE")
    stitch_calls = []

    result = perception_capture.recover_orphan_diagnostic_video(
        tmp_path,
        video_validator=lambda _path: 2,
        video_stitcher=lambda *_args: stitch_calls.append(True),
    )

    assert stitch_calls == []
    assert result["status"] == "recovered_video"
    assert (tmp_path / "rgb.recovered.mp4").read_bytes() == b"ALREADY-COMPLETE"
    assert not recovering.exists()


def test_segment_validator_abort_drains_full_queue_and_joins_worker():
    entered = threading.Event()
    release = threading.Event()

    def validate(_path):  # noqa: ANN001
        entered.set()
        assert release.wait(2.0)
        return 1

    worker = perception_capture._SegmentValidationWorker(
        validate,
        lambda _path: {},
        negotiated_mode=None,
        capacity=1,
    )
    payload = {
        "segment_index": 0,
        "path": "/tmp/segment.mp4",
        "relative_path": "rgb_segments/segment_000000.mp4",
        "frame_count": 1,
        "sha256": "a" * 64,
        "bytes": 1,
    }
    worker.submit(payload)
    assert entered.wait(1.0)
    worker.submit({**payload, "segment_index": 1})
    outcome = {}
    aborter = threading.Thread(
        target=lambda: outcome.setdefault("result", worker.abort(timeout_s=2.0)),
        daemon=True,
    )
    try:
        aborter.start()
        time.sleep(0.03)
        release.set()
        aborter.join(2.0)
        assert not aborter.is_alive()
        assert worker._thread.is_alive() is False
        assert outcome["result"] is True
    finally:
        release.set()
        if worker._thread.is_alive():
            worker.abort(timeout_s=1.0)


def test_non_monotonic_camera_host_time_aborts_instead_of_sealing_bad_timeline(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    first = _frame(11)
    capture.write_frame(first, _tracking(first))
    second = _frame(12)
    second.t = first.t

    with pytest.raises(RuntimeError, match="timestamp did not increase"):
        capture.write_frame(second, _tracking(second))

    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "diagnostic_aborted"
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_abort_preserves_partial_artifacts_but_never_writes_a_completion_seal(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(3)
    capture.write_frame(frame, _tracking(frame))

    capture.abort("operator disconnected")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["abort_reason"] == "operator disconnected"
    assert manifest["training_eligible"] is False
    assert any(path.name.startswith("rgb") for path in tmp_path.iterdir())
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert not (tmp_path / "FINALIZATION.json").exists()


def test_depth_write_failure_seals_the_exact_physical_video_prefix_and_abort_is_idempotent(
    tmp_path, monkeypatch,
):
    capture, writers, _ = _capture(tmp_path, video_segment_frames=10)
    capture.start()
    original_save = perception_capture._atomic_save_npy
    calls = 0

    def fail_second_depth(path, array):  # noqa: ANN001
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected depth fsync failure")
        return original_save(path, array)

    monkeypatch.setattr(perception_capture, "_atomic_save_npy", fail_second_depth)
    first = _frame(1)
    second = _frame(2)
    capture.write_frame(first, _tracking(first))

    with pytest.raises(OSError, match="injected depth fsync failure"):
        capture.write_frame(second, _tracking(second))

    journal = [
        json.loads(line)
        for line in (tmp_path / "video_segments.jsonl").read_text().splitlines()
    ]
    assert len(writers[0].frames) == 2
    assert journal[0]["frame_count"] == 2
    assert journal[0]["first_frame_index"] == 1
    assert journal[0]["last_frame_index"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["n_rgb_frames"] == manifest["n_depth_frames"] == 1
    first_reason = manifest["abort_reason"]

    capture.abort("second abort must not replace first evidence")

    assert json.loads((tmp_path / "manifest.json").read_text())["abort_reason"] == first_reason
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_depth_identity_tamper_fails_finalization_without_completion_seal(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    for index in (1, 2):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))
    with (tmp_path / "depth" / "depth_000001.npy.zlib").open("ab") as fp:
        fp.write(b"tamper")

    with pytest.raises(RuntimeError, match="depth frame identity changed"):
        capture.finalize()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_recording_aborts_before_end_when_final_rgb_copy_reserve_is_unavailable(tmp_path):
    # Start itself has enough room for the configured safety watermark.  As soon
    # as video bytes exist, however, the recorder must also retain enough free
    # space for the conventional rgb.mp4 that End creates beside the segments.
    capture, _, _ = _capture(
        tmp_path, free_bytes=1025, require_depth=False, video_segment_frames=1,
    )
    capture.start()
    frame = _frame(1)
    frame.depth = None

    with pytest.raises(RuntimeError, match=r"final rgb\.mp4.*reserve"):
        capture.write_frame(frame, _tracking(frame))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "diagnostic_aborted"
    assert manifest["training_eligible"] is False
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_final_index_validation_streams_jsonl_instead_of_reading_it_whole(
    tmp_path, monkeypatch,
):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    for index in range(1, 6):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    original_read_text = Path.read_text

    def bounded_read_text(path, *args, **kwargs):  # noqa: ANN001
        if Path(path).name in {"tracker.jsonl", "inference.jsonl"}:
            raise AssertionError("final index validation must stream JSONL")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", bounded_read_text)

    result = capture.finalize()

    assert result["n_rgb_frames"] == 5
    assert result["decoded_rgb_frames"] == 5


def test_recording_boundary_can_be_bound_once_after_slow_file_initialization(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    boundary = {
        "start_host_t_s": 123.0,
        "camera_start_boundary_index": 77,
        "camera_delivery_interval_id": 4,
    }

    capture.set_recording_boundary(boundary)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["recording_boundary"] == boundary
    with pytest.raises(RuntimeError, match="already bound"):
        capture.set_recording_boundary({**boundary, "camera_start_boundary_index": 78})
    capture.abort("test cleanup")


def test_finalization_verifies_encoded_fps_shape_and_source_timeline(tmp_path):
    provenance = json.loads(json.dumps(PROVENANCE))
    provenance["camera"]["negotiated_mode"] = {
        "width": 12, "height": 8, "fps": 30.0,
    }
    capture, _, _ = _capture(
        tmp_path,
        provenance=provenance,
        video_mode_probe=lambda _path: {"fps": 30.0, "width": 12, "height": 8},
    )
    capture.start()
    for index in (1, 2, 3):
        frame = _frame(index)
        capture.write_frame(frame, _tracking(frame))

    result = capture.finalize()

    proof = result["video_timebase"]
    assert proof["verified"] is True
    assert proof["encoded_fps"] == proof["negotiated_fps"] == 30.0
    assert (proof["encoded_width"], proof["encoded_height"]) == (12, 8)
    seal = json.loads((tmp_path / "DIAGNOSTIC_FINALIZATION.json").read_text())
    assert seal["video_timebase"] == proof
    assert seal["storage"]["depth_format"] == "formal_per_frame_zlib_npy_v1"
    assert seal["storage"]["estimated_remaining_seconds"] > 0


@pytest.mark.parametrize(
    ("probe", "timestamp_step", "message"),
    [
        ({"fps": 20.0, "width": 12, "height": 8}, 1 / 30, "encoded FPS"),
        ({"fps": 30.0, "width": 10, "height": 8}, 1 / 30, "encoded shape"),
        ({"fps": 30.0, "width": 12, "height": 8}, 1 / 11, "observed FPS"),
    ],
)
def test_finalization_rejects_video_or_source_timebase_drift(
    tmp_path, probe, timestamp_step, message,
):
    provenance = json.loads(json.dumps(PROVENANCE))
    provenance["camera"]["negotiated_mode"] = {
        "width": 12, "height": 8, "fps": 30.0,
    }
    capture, _, _ = _capture(
        tmp_path,
        provenance=provenance,
        video_mode_probe=lambda _path: probe,
    )
    capture.start()
    for index in (1, 2, 3):
        frame = _frame(index)
        frame.t = 100.0 + index * timestamp_step
        tracking = _tracking(frame)
        capture.write_frame(frame, tracking)

    with pytest.raises(RuntimeError, match=message):
        capture.finalize()

    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())[
        "status"
    ] == "diagnostic_aborted"


def test_depth_dtype_is_uint16_before_space_budget_or_persistence(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(1)
    frame.depth = frame.depth.astype(np.float32)

    with pytest.raises(RuntimeError, match="invalid depth"):
        capture.write_frame(frame, _tracking(frame))

    assert not list((tmp_path / "depth").glob("*.npy.zlib"))
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()


def test_all_zero_depth_aborts_before_persistence(tmp_path):
    capture, _, _ = _capture(tmp_path)
    capture.start()
    frame = _frame(1)
    frame.depth[:] = 0

    with pytest.raises(RuntimeError, match="no valid depth"):
        capture.write_frame(frame, _tracking(frame))

    assert not list((tmp_path / "depth").glob("*.npy.zlib"))
    assert not (tmp_path / "DIAGNOSTIC_FINALIZATION.json").exists()
