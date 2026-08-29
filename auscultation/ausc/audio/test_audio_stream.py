"""Audio capture clock tests that do not open a sound device."""

from __future__ import annotations

import json
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from auscultation.ausc.audio import audio_stream as audio


def test_callback_uses_portaudio_adc_time_instead_of_callback_arrival(monkeypatch):
    stream = audio.AudioStream(0, sr_in=48_000, sr_model=16_000, channels=1,
                               buffer_seconds=1.0, block_ms=100.0)
    block = np.ones((4_800, 1), dtype=np.float32)
    # PortAudio says this block started at 20.0, its clock is now 20.2, and the
    # Python callback arrived at host monotonic 100.0.  The final sample is thus
    # 100.0 - 0.2 + 0.1 = 99.9, not callback-arrival time 100.0.
    ti = SimpleNamespace(inputBufferAdcTime=20.0, currentTime=20.2)
    monkeypatch.setattr(audio.time, "monotonic", lambda: 100.0)

    stream._callback(block, len(block), ti, None)

    win = stream.latest_window(0.1)
    assert win.t_end == pytest.approx(99.9, abs=1e-6)


def test_capture_spools_to_disk_with_bounded_memory_and_publishes_exact_wav(tmp_path):
    stream = audio.AudioStream(
        0, sr_in=48, sr_model=16, channels=1,
        buffer_seconds=0.25, block_ms=250.0,
    )
    stream.begin_recording(tmp_path, t0=10.0)

    # The recording duration is deliberately absent from the API: retained
    # memory is bounded by the fixed callback queue/ring, not by session length.
    assert stream.recording_queue_capacity <= 128
    assert stream._ring._cap == 12

    stream._ingest(np.linspace(-0.8, 0.8, 48, dtype=np.float32), 11.0)
    stream._ingest(np.linspace(-0.4, 0.4, 48, dtype=np.float32), 12.0)
    summary = stream.end_recording(t_end=12.0)

    assert not hasattr(summary, "samples")
    assert summary.status == audio.STATUS_OK
    assert summary.sr == 16
    assert summary.n_samples == 32
    assert summary.t_start == pytest.approx(10.0)
    assert summary.t_end == pytest.approx(12.0)
    assert len(summary.sha256) == 64
    with wave.open(str(tmp_path / "audio.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16)
        assert wav.getnframes() == 32
    rows = [json.loads(line) for line in (tmp_path / "audio_times.jsonl").read_text().splitlines()]
    assert sum(row["sample_count"] for row in rows) == 32
    assert not (tmp_path / "audio.wav.tmp").exists()


def _record_pcm_signal(tmp_path, pcm: np.ndarray):
    sample_rate = 16_000
    values = np.asarray(pcm, dtype=np.float32).reshape(-1)
    stream = audio.AudioStream(
        0, sr_in=sample_rate, sr_model=sample_rate, channels=1,
        buffer_seconds=0.25, block_ms=50.0,
    )
    stream.begin_recording(tmp_path, t0=10.0)
    stream._ingest(values, 10.0 + values.size / sample_rate)
    return stream.end_recording(t_end=10.0 + values.size / sample_rate)


@pytest.mark.parametrize(
    ("sample_value", "reason"),
    [
        (0.0, "all-zero"),
        (0.125, "constant PCM"),
    ],
)
def test_audio_authenticity_rejects_all_zero_and_constant_pcm(
    tmp_path, sample_value, reason,
):
    signal = np.full(32_000, sample_value, dtype=np.float32)

    with pytest.raises(RuntimeError, match=f"authenticity.*{reason}"):
        _record_pcm_signal(tmp_path, signal)

    assert not (tmp_path / "audio.wav").exists()
    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    assert metadata["status"] == "interrupted"


def test_frozen_dropout_degrades_with_region_ledger_but_preserves_the_session(tmp_path):
    rng = np.random.default_rng(71)
    one_block = rng.integers(-2_000, 2_001, size=800).astype(np.float32) / 32_767.0
    signal = np.tile(one_block, 40)

    summary = _record_pcm_signal(tmp_path, signal)

    assert (tmp_path / "audio.wav").exists()
    receipt = summary.signal_authenticity
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["repeated_frozen_blocks"]
    assert receipt["checks"]["repeated_frozen_blocks"] == "DEGRADED"
    regions = receipt["evidence"]["frozen_regions"]
    assert regions == [{
        "start_sample": 0,
        "start_s": 0.0,
        "n_blocks": 40,
        "duration_s": 2.0,
    }]
    assert receipt["evidence"]["frozen_regions_truncated"] is False


def test_lsb_scale_random_noise_degrades_instead_of_voiding_the_session(tmp_path):
    pcm = np.tile(np.asarray([-1, 0, 1], dtype=np.int16), 16_000)
    signal = pcm.astype(np.float32) / 32_767.0

    summary = _record_pcm_signal(tmp_path, signal)

    assert (tmp_path / "audio.wav").exists()
    receipt = summary.signal_authenticity
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["minimum_effective_ac_energy"]
    assert receipt["checks"]["minimum_effective_ac_energy"] == "DEGRADED"


def test_severe_clipping_degrades_instead_of_voiding_the_session(tmp_path):
    signal = np.tile(np.asarray([-1.0, 1.0], dtype=np.float32), 16_000)

    summary = _record_pcm_signal(tmp_path, signal)

    assert (tmp_path / "audio.wav").exists()
    receipt = summary.signal_authenticity
    # The period-2 square wave also repeats bit-identical 50 ms blocks, so the
    # frozen-block ledger fires alongside clipping; both stay non-fatal.
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["severe_clipping", "repeated_frozen_blocks"]
    assert receipt["checks"]["severe_clipping"] == "DEGRADED"


def test_severe_dc_offset_degrades_instead_of_voiding_the_session(tmp_path):
    phase = np.arange(32_000, dtype=np.float32)
    signal = 0.5 + 0.01 * np.sin(2.0 * np.pi * phase / 127.0)

    summary = _record_pcm_signal(tmp_path, signal)

    assert (tmp_path / "audio.wav").exists()
    receipt = summary.signal_authenticity
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["severe_dc"]
    assert receipt["checks"]["severe_dc"] == "DEGRADED"


def test_quiet_varying_audio_is_not_misclassified_and_receipt_is_versioned(tmp_path):
    rng = np.random.default_rng(17)
    pcm = rng.integers(-8, 9, size=32_000, dtype=np.int16)
    signal = pcm.astype(np.float32) / 32_767.0

    summary = _record_pcm_signal(tmp_path, signal)

    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    receipt = metadata["signal_authenticity"]
    assert receipt["schema_version"] == "wam_audio_signal_authenticity_v1"
    assert receipt["policy_version"] == "audio_signal_degraded_real_v3"
    assert receipt["status"] == "PASS"
    assert receipt["degradations"] == []
    assert receipt["evidence"]["frozen_regions"] == []
    assert receipt["checks"] == {
        "all_zero": "PASS",
        "constant_pcm": "PASS",
        "minimum_effective_ac_energy": "PASS",
        "repeated_frozen_blocks": "PASS",
        "severe_clipping": "PASS",
        "severe_dc": "PASS",
    }
    assert summary.signal_authenticity == receipt
    assert summary.peak_db < -65.0


def test_partial_audio_recovery_is_diagnostic_only_and_preserves_raw_bytes(tmp_path):
    (tmp_path / "audio_capture.json").write_text(json.dumps({
        "schema_version": "wam_audio_stream_capture_v1",
        "status": "recording",
        "training_eligible": False,
        "sample_rate": 16,
        "t0_monotonic": 20.0,
    }))
    pcm = np.asarray([100, -200, 300, -400], dtype="<i2").tobytes()
    (tmp_path / "audio.partial.pcm").write_bytes(pcm + b"x")

    result = audio.recover_partial_recording(tmp_path)

    assert result["status"] == "diagnostic_recovered"
    assert result["training_eligible"] is False
    assert result["n_samples"] == 4
    assert (tmp_path / "audio.partial.pcm").read_bytes() == pcm + b"x"
    assert not (tmp_path / "audio.wav").exists()
    with wave.open(str(tmp_path / "audio.recovered.wav"), "rb") as wav:
        assert wav.getframerate() == 16
        assert wav.getnframes() == 4


def test_streaming_resampler_is_block_invariant_and_rejects_out_of_band_alias():
    sr_in, sr_out = 48_000, 16_000
    t = np.arange(sr_in // 10, dtype=np.float32) / sr_in
    source = (
        0.5 * np.sin(2 * np.pi * 1_000 * t)
        + 0.5 * np.sin(2 * np.pi * 12_000 * t)
    ).astype(np.float32)

    whole = audio._StreamingResampler(sr_in, sr_out)
    expected = np.concatenate((whole.push(source), whole.finish()))
    chunked = audio._StreamingResampler(sr_in, sr_out)
    actual = np.concatenate([
        chunked.push(source[:713]),
        chunked.push(source[713:2_401]),
        chunked.push(source[2_401:]),
        chunked.finish(),
    ])

    assert actual.shape == expected.shape == (sr_out // 10,)
    assert np.allclose(actual, expected, atol=1e-6)
    spectrum = np.abs(np.fft.rfft(actual))
    signal_bin = int(round(1_000 * actual.size / sr_out))
    alias_bin = int(round(4_000 * actual.size / sr_out))
    assert spectrum[alias_bin] < 0.05 * spectrum[signal_bin]


def test_portaudio_overflow_aborts_formal_audio_instead_of_sealing_gap(tmp_path):
    stream = audio.AudioStream(
        0, sr_in=48, sr_model=16, channels=1,
        buffer_seconds=0.25, block_ms=250.0,
    )
    stream.begin_recording(tmp_path, t0=10.0)
    stream._ingest(
        np.ones(48, dtype=np.float32), 11.0,
        callback_status="input overflow",
    )

    with pytest.raises(RuntimeError, match="PortAudio callback.*overflow"):
        stream.end_recording(t_end=11.0)

    assert not (tmp_path / "audio.wav").exists()
    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["training_eligible"] is False


def test_timestamp_gap_and_post_end_block_cannot_be_sealed(tmp_path):
    stream = audio.AudioStream(
        0, sr_in=48, sr_model=16, channels=1,
        buffer_seconds=0.25, block_ms=250.0,
    )
    stream.begin_recording(tmp_path, t0=10.0)
    stream._ingest(np.ones(48, dtype=np.float32), 11.0)
    stream._ingest(np.ones(48, dtype=np.float32), 20.0)

    with pytest.raises(RuntimeError, match="audio input timeline discontinuity"):
        stream.end_recording(t_end=12.0)

    assert not (tmp_path / "audio.wav").exists()
    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["training_eligible"] is False


def test_audio_health_peak_is_computed_only_from_samples_before_end(tmp_path):
    stream = audio.AudioStream(
        0, sr_in=48, sr_model=16, channels=1,
        buffer_seconds=0.25, block_ms=250.0,
    )
    stream.begin_recording(tmp_path, t0=10.0)
    stream._ingest(np.linspace(-0.05, 0.05, 48, dtype=np.float32), 11.0)
    stream._ingest(np.ones(48, dtype=np.float32), 12.0)

    summary = stream.end_recording(t_end=11.0)

    assert summary.n_samples == 16
    assert summary.peak_db < -20.0


def test_live_monitor_returns_only_new_samples_and_reports_ring_overrun():
    stream = audio.AudioStream(
        0, sr_in=10, sr_model=5, channels=1,
        buffer_seconds=1.0, block_ms=100.0,
    )
    stream._ingest(np.arange(8, dtype=np.float32) / 10.0, 10.0)

    first = stream.monitor_chunk(None, max_samples=4)

    assert first["sample_rate"] == 10
    assert first["start_sample"] == 4
    assert first["next_sample"] == 8
    assert first["overrun"] is False
    assert first["samples"].tolist() == pytest.approx([0.4, 0.5, 0.6, 0.7])

    stream._ingest(np.arange(8, 16, dtype=np.float32) / 20.0, 10.8)
    second = stream.monitor_chunk(first["next_sample"], max_samples=16)

    assert second["start_sample"] == 8
    assert second["next_sample"] == 16
    assert second["overrun"] is False
    assert second["samples"].tolist() == pytest.approx(
        [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    )

    # The requested cursor has fallen behind the 10-sample bounded ring. The
    # stream resumes from the oldest retained sample and tells the browser to
    # reset its playback schedule instead of pretending the gap did not exist.
    overrun = stream.monitor_chunk(0, max_samples=16)
    assert overrun["start_sample"] == 6
    assert overrun["next_sample"] == 16
    assert overrun["overrun"] is True
    assert len(overrun["samples"]) == 10


def test_stop_aborts_active_spool_and_closes_device_even_when_stop_raises(tmp_path):
    stream = audio.AudioStream(0, sr_in=48, sr_model=16, channels=1)
    stream.begin_recording(tmp_path, t0=10.0)

    class Device:
        closed = False

        def stop(self):
            raise RuntimeError("device stop failed")

        def close(self):
            self.closed = True

    device = Device()
    stream._stream = device

    with pytest.raises(RuntimeError, match="device stop failed"):
        stream.stop()

    assert device.closed is True
    assert stream._stream is None
    assert stream._recording is None
    metadata = json.loads((tmp_path / "audio_capture.json").read_text())
    assert metadata["status"] == "interrupted"


def test_start_closes_candidate_device_when_portaudio_start_fails(monkeypatch):
    class Device:
        closed = False

        def start(self):
            raise RuntimeError("PortAudio start failed")

        def close(self):
            self.closed = True

    device = Device()
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=lambda **_kwargs: device),
    )
    stream = audio.AudioStream(0, sr_in=48, sr_model=16, channels=1)

    with pytest.raises(RuntimeError, match="PortAudio start failed"):
        stream.start()

    assert device.closed is True
    assert stream._stream is None
