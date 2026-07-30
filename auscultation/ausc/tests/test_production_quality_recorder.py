"""Contracts for the production quality session recorder (write-side composition)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.production_quality_recorder import (
    ProductionQualityRecorderError,
    ProductionQualitySessionRecorder,
)
from ausc.realtime_q import read_quality_packet_log, verify_quality_packet_log

# The production Q chain binds to force.csv, which the force channel writes in a
# real session; reuse the real force recorder to emit a matching one here.
_FORCE_FEEDBACK = Path(__file__).resolve().parents[2] / "mechanical_sensor" / "force_feedback"
sys.path.insert(0, str(_FORCE_FEEDBACK))
import force_capture  # noqa: E402
import force_sensor  # noqa: E402

_MODEL_SHA = "a" * 64
_MODEL_RELEASE = "clinical-q-capture-test-" + _MODEL_SHA[:20]
_CAL_SHA = "c" * 64
_POLICY = {
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
            "release_id": "clinical-q-calibration-test-" + _CAL_SHA[:20],
            "sha256": _CAL_SHA,
        },
        "transform_to_p_usable": "identity",
        "output_range": [0.0, 1.0],
    },
}

_FRAMES = ((30, 0.1, 7.25), (31, 0.2, 7.55), (32, 0.3, 7.85))


def _scorer(samples, _sample_rate):
    return float(np.mean(samples))


def _recorder(tmp_path: Path) -> ProductionQualitySessionRecorder:
    return ProductionQualitySessionRecorder(
        tmp_path,
        scorer=_scorer,
        model_release_id=_MODEL_RELEASE,
        model_sha256=_MODEL_SHA,
        policy=_POLICY,
    )


def _audio_window(sequence: int, value: float, inference_t: float) -> SimpleNamespace:
    samples = np.linspace(value, value + 0.01, 4_000, dtype=np.float32)
    window_index = sequence - 30
    return SimpleNamespace(
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
    )


def _force_sample(sequence: int, inference_t: float) -> SimpleNamespace:
    return SimpleNamespace(
        t=inference_t - 0.05,
        raw=31_000 + (sequence - 30),
        hall_v=1.56 + (sequence - 30) * 0.001,
        displacement_mm=1.20 + (sequence - 30) * 0.01,
        force_n=4.0,
        status="OK",
        device_t_ms=100.0 + (sequence - 30) * 300.0,
    )


def _write_force_csv(tmp_path: Path) -> None:
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
    recorder.finish(t_end=7.8)


def _record_all(recorder: ProductionQualitySessionRecorder) -> None:
    for sequence, value, inference_t in _FRAMES:
        recorder.record_window(
            audio_window=_audio_window(sequence, value, inference_t),
            force_sample=_force_sample(sequence, inference_t),
            inference_sequence=sequence,
            inference_monotonic_s=inference_t,
            observed_monotonic_s=inference_t + 0.01,
        )


def test_recorder_publishes_a_verifiable_quality_session(tmp_path):
    recorder = _recorder(tmp_path)
    _record_all(recorder)
    _write_force_csv(tmp_path)
    assert recorder.window_count == 3
    recorder.finalize(now_monotonic_s=7.9)
    assert recorder.finalized is True

    assert (tmp_path / "quality_inference_packets.jsonl").is_file()
    assert (tmp_path / "quality_inference_authenticity.json").is_file()

    # The full Q chain verifies, and every packet carries raw_source + model_input.
    receipt = verify_quality_packet_log(tmp_path)
    assert receipt["packet_count"] == 3
    packets = read_quality_packet_log(tmp_path / "quality_inference_packets.jsonl")
    for packet in packets:
        assert "raw_source" in packet["audio_window"]
        assert "model_input" in packet["audio_window"]


def test_recorded_quality_session_is_replayable_by_the_runner(tmp_path):
    from ausc.perception.replay_runner import reconstruct_quality_items

    recorder = _recorder(tmp_path)
    _record_all(recorder)
    _write_force_csv(tmp_path)
    recorder.finalize(now_monotonic_s=7.9)

    items, identity = reconstruct_quality_items(tmp_path)
    assert len(items) == 3
    assert identity == {"release_id": _MODEL_RELEASE, "model_sha256": _MODEL_SHA}
    assert items[0].model_input.window.shape[0] > 0
    assert set(items[0].recorded_outputs) == {"q_raw"}


def test_recorder_is_not_ready_and_refuses_receipt_without_enough_packets(tmp_path):
    recorder = _recorder(tmp_path)
    sequence, value, inference_t = _FRAMES[0]
    recorder.record_window(
        audio_window=_audio_window(sequence, value, inference_t),
        force_sample=_force_sample(sequence, inference_t),
        inference_sequence=sequence,
        inference_monotonic_s=inference_t,
        observed_monotonic_s=inference_t + 0.01,
    )
    # min_packets_for_receipt is 2; a single-packet session cannot finalize.
    with pytest.raises(Exception):
        recorder.finalize(now_monotonic_s=inference_t + 0.05)


def test_recorder_refuses_windows_after_finalize(tmp_path):
    recorder = _recorder(tmp_path)
    _record_all(recorder)
    _write_force_csv(tmp_path)
    recorder.finalize(now_monotonic_s=7.9)
    sequence, value, inference_t = (33, 0.4, 8.15)
    with pytest.raises(ProductionQualityRecorderError, match="already finalized"):
        recorder.record_window(
            audio_window=_audio_window(sequence, value, inference_t),
            force_sample=_force_sample(sequence, inference_t),
            inference_sequence=sequence,
            inference_monotonic_s=inference_t,
            observed_monotonic_s=inference_t + 0.01,
        )


def test_recorder_finalize_is_single_shot(tmp_path):
    recorder = _recorder(tmp_path)
    _record_all(recorder)
    _write_force_csv(tmp_path)
    recorder.finalize(now_monotonic_s=7.9)
    with pytest.raises(ProductionQualityRecorderError, match="already finalized"):
        recorder.finalize(now_monotonic_s=7.95)


def test_recorder_rejects_a_non_callable_scorer(tmp_path):
    with pytest.raises(ProductionQualityRecorderError, match="not callable"):
        ProductionQualitySessionRecorder(
            tmp_path, scorer=object(), model_release_id=_MODEL_RELEASE,
            model_sha256=_MODEL_SHA, policy=_POLICY,
        )
