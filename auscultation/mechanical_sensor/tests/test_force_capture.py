"""Fail-closed Hall-force capture tests; no serial hardware is opened."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

FF = Path(__file__).resolve().parents[1] / "force_feedback"
sys.path.insert(0, str(FF))

import force_capture as capture  # noqa: E402
import force_sensor as fs  # noqa: E402


CALIBRATION = {
    "sha256": "c" * 64,
    "calibrated": True,
    "source_name": "force_feedback_calibration_test.json",
}
DEVICE = {
    "port": "/dev/serial/by-id/usb-test-hall",
    "baud": 115200,
    "usb_vid": "2e8a",
    "usb_pid": "0005",
    "serial_number": "TEST-HALL-01",
}


def _sample(
    host_t_s: float,
    device_t_ms: float | None,
    *,
    raw: int = 31_000,
    hall_v: float = 1.561,
    displacement_mm: float = 1.25,
    force_n: float = 2.75,
    status: str = fs.STATUS_WARN,
) -> fs.ForceSample:
    return fs.ForceSample(
        t=host_t_s,
        raw=raw,
        hall_v=hall_v,
        displacement_mm=displacement_mm,
        force_n=force_n,
        status=status,
        device_t_ms=device_t_ms,
    )


def _recorder(tmp_path: Path, **kwargs) -> capture.ForceCaptureRecorder:
    options = {
        "t0": 10.0,
        "calibration_identity": CALIBRATION,
        "device_identity": DEVICE,
        "queue_samples": 32,
        "checkpoint_samples": 8,
        "checkpoint_interval_s": 60.0,
        "max_device_gap_ms": 25.0,
        "max_host_gap_ms": 25.0,
    }
    options.update(kwargs)
    return capture.ForceCaptureRecorder(tmp_path, **options)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pinned_calibration_artifact() -> tuple[bytes, dict, fs.ForceCalibration]:
    payload = json.loads(
        (FF / "force_feedback_calibration_veo6.json").read_text(encoding="utf-8")
    )
    payload["calibrated"] = True
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    sha256 = hashlib.sha256(raw).hexdigest()
    identity = {
        "sha256": sha256,
        "calibrated": True,
        "source_name": "force_feedback_calibration_test.json",
        "artifact_schema_version": "wam_force_calibration_json_v1",
        "release_id": f"hall-force-calibration-test-{sha256[:20]}",
    }
    return raw, identity, fs.ForceCalibration.from_bytes(raw)


def test_exact_start_and_end_timestamp_contract(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for sample in (
        _sample(9.99, 90.0),
        _sample(10.00, 100.0),
        _sample(10.01, 110.0),
        _sample(10.02, 120.0),
        _sample(10.03, 130.0),
    ):
        recorder.submit(sample)

    summary = recorder.finish(t_end=10.02)

    rows = _csv_rows(tmp_path / "force.csv")
    assert [float(row["host_t_s"]) for row in rows] == pytest.approx([10.0, 10.01, 10.02])
    assert [float(row["device_t_ms"]) for row in rows] == pytest.approx([100.0, 110.0, 120.0])
    assert summary.n_samples == 3
    assert summary.t_start == pytest.approx(10.0)
    assert summary.t_end == pytest.approx(10.02)
    assert len(summary.sha256) == 64

    metadata = json.loads((tmp_path / "force_capture.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "sensors_finalized"
    assert metadata["training_eligible"] is False
    assert metadata["n_samples"] == 3
    assert metadata["csv_sha256"] == summary.sha256
    assert metadata["calibration"] == CALIBRATION
    assert metadata["device_identity"] == DEVICE
    assert metadata["t0_monotonic"] == pytest.approx(10.0)
    assert metadata["requested_t_end_monotonic"] == pytest.approx(10.02)
    assert (tmp_path / capture.COMPLETE_NAME).is_file()
    assert capture.verify_force_capture(tmp_path).n_samples == 3


def test_all_zero_force_signal_fails_closed_without_completion_marker(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(3):
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=0,
            hall_v=0.0,
            displacement_mm=0.0,
            force_n=0.0,
        ))

    with pytest.raises(capture.ForceCaptureError, match="authenticity.*all-zero"):
        recorder.finish(t_end=10.02)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_short_near_zero_force_signal_degrades_instead_of_failing(
    tmp_path: Path,
) -> None:
    # 2026-08-29 (operator-approved, P0-4): an unloaded advisory Hall channel
    # is a real capture state, not a session-voiding fault.
    recorder = _recorder(tmp_path)
    recorder.start()
    for index, raw in enumerate((30_000, 30_001, 30_002)):
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=raw,
            force_n=0.001 + index * 0.0001,
        ))

    summary = recorder.finish(t_end=10.02)

    receipt = summary.signal_authenticity
    assert receipt["status"] == "INSUFFICIENT_EVIDENCE"
    assert "near_zero_response" in receipt["degradations"]
    assert receipt["checks"]["near_zero_response"] == "DEGRADED"
    assert (tmp_path / capture.COMPLETE_NAME).exists()


def test_short_constant_nonzero_force_is_insufficient_for_promotion(
    tmp_path: Path,
) -> None:
    calibration_raw, identity, calibration = _pinned_calibration_artifact()
    recorder = _recorder(
        tmp_path,
        calibration_identity=identity,
        calibration_artifact=calibration_raw,
    )
    recorder.start()
    for index in range(51):
        recorder.submit(calibration.sample_from_raw(
            31_000,
            10.0 + index * 0.01,
            device_t_ms=100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()
    summary = recorder.finish(t_end=10.5)

    assert summary.signal_authenticity["status"] == "INSUFFICIENT_EVIDENCE"
    assert summary.signal_authenticity["promotion_eligible"] is False
    assert summary.signal_authenticity["checks"]["minimum_evidence"] == (
        "INSUFFICIENT_EVIDENCE"
    )
    assert "minimum duration/sample evidence" in summary.signal_authenticity[
        "promotion_blockers"
    ]


def test_all_zero_raw_adc_fails_even_if_recorded_force_is_nonzero(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(3):
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=0,
            force_n=1.5,
        ))

    with pytest.raises(capture.ForceCaptureError, match="all-zero raw ADC"):
        recorder.finish(t_end=10.02)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_long_bit_identical_force_degrades_and_short_steady_is_not_promotable(
    tmp_path: Path,
) -> None:
    short_session = tmp_path / "short-steady"
    short = _recorder(short_session)
    short.start()
    for index in range(101):
        short.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            short._queue.join()
    short_summary = short.finish(t_end=11.0)

    assert short_summary.status == "OK"
    assert short_summary.signal_authenticity["status"] == "INSUFFICIENT_EVIDENCE"
    assert short_summary.signal_authenticity["promotion_eligible"] is False

    frozen_session = tmp_path / "long-frozen"
    frozen = _recorder(frozen_session)
    frozen.start()
    for index in range(601):
        frozen.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
        ))
        if index % 16 == 15:
            frozen._queue.join()

    frozen_summary = frozen.finish(t_end=16.0)

    receipt = frozen_summary.signal_authenticity
    assert receipt["status"] == "PASS_DEGRADED"
    assert "digital_freeze" in receipt["degradations"]
    assert receipt["checks"]["digital_freeze"] == "DEGRADED"
    (region,) = receipt["evidence"]["frozen_regions"]
    assert region["n_samples"] == 601
    assert region["duration_s"] == pytest.approx(6.0, abs=0.02)
    assert (frozen_session / capture.COMPLETE_NAME).exists()


def test_force_conversion_pinned_while_raw_adc_moves_fails_closed(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(601):
        raw = 30_000 + (index % 201)
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=raw,
            hall_v=raw / 20_000.0,
            displacement_mm=(raw - 30_000) / 100.0,
            force_n=2.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()

    with pytest.raises(capture.ForceCaptureError, match="authenticity.*conversion.*pinned"):
        recorder.finish(t_end=16.0)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_dynamic_force_signal_publishes_versioned_authenticity_receipt(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(601):
        raw = 30_000 + (index % 201)
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=raw,
            hall_v=raw / 20_000.0,
            displacement_mm=(raw - 30_000) / 100.0,
            force_n=(raw - 30_000) / 40.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()

    summary = recorder.finish(t_end=16.0)

    metadata = json.loads((tmp_path / "force_capture.json").read_text())
    receipt = metadata["signal_authenticity"]
    assert receipt["schema_version"] == "wam_force_signal_authenticity_v1"
    assert receipt["policy_version"] == "force_signal_degraded_real_v4"
    assert receipt["status"] == "PASS"
    assert receipt["degradations"] == []
    assert receipt["evidence"]["frozen_regions"] == []
    assert receipt["promotion_eligible"] is False
    assert receipt["checks"] == {
        "all_zero": "PASS",
        "calibration_recomputation": "NOT_AVAILABLE",
        "conversion_response": "PASS",
        "digital_freeze": "PASS",
        "minimum_evidence": "PASS",
        "near_constant_response": "PASS",
        "near_zero_response": "PASS",
    }
    assert summary.signal_authenticity == receipt
    assert capture.verify_force_capture(tmp_path).signal_authenticity == receipt


def test_long_near_constant_force_response_degrades_with_ledger(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(601):
        offset = index % 2
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=31_000 + offset,
            hall_v=1.561 + offset * 0.00001,
            displacement_mm=1.25 + offset * 0.0001,
            force_n=2.75 + offset * 0.0001,
        ))
        if index % 16 == 15:
            recorder._queue.join()

    summary = recorder.finish(t_end=16.0)

    receipt = summary.signal_authenticity
    assert receipt["status"] == "PASS_DEGRADED"
    assert receipt["degradations"] == ["near_constant_response"]
    assert receipt["checks"]["near_constant_response"] == "DEGRADED"
    assert receipt["evidence"]["frozen_regions"] == []
    assert (tmp_path / capture.COMPLETE_NAME).exists()


def test_calibration_artifact_is_persisted_and_every_row_is_recomputed(
    tmp_path: Path,
) -> None:
    calibration_raw, identity, calibration = _pinned_calibration_artifact()
    recorder = _recorder(
        tmp_path,
        calibration_identity=identity,
        calibration_artifact=calibration_raw,
    )
    recorder.start()
    for index, raw in enumerate((30_000, 30_050, 30_100)):
        recorder.submit(calibration.sample_from_raw(
            raw,
            10.0 + index * 0.01,
            device_t_ms=100.0 + index * 10.0,
        ))

    summary = recorder.finish(t_end=10.02)
    artifact_path = tmp_path / "force_calibration.json"
    assert artifact_path.read_bytes() == calibration_raw
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == identity["sha256"]
    assert summary.signal_authenticity["checks"]["calibration_recomputation"] == "PASS"
    assert capture.verify_force_capture(tmp_path).signal_authenticity == (
        summary.signal_authenticity
    )


def test_finalized_force_capture_rejects_reappeared_staging_artifact(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index in range(3):
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=31_000 + index,
        ))
    recorder.finish(t_end=10.02)
    (tmp_path / capture.CSV_INPROGRESS_NAME).write_text("late partial\n")

    with pytest.raises(
        capture.ForceCaptureVerificationError,
        match="staging|in-progress",
    ):
        capture.verify_force_capture(tmp_path)


def test_inconsistent_raw_to_force_conversion_cannot_seal_with_pinned_calibration(
    tmp_path: Path,
) -> None:
    calibration_raw, identity, calibration = _pinned_calibration_artifact()
    recorder = _recorder(
        tmp_path,
        calibration_identity=identity,
        calibration_artifact=calibration_raw,
    )
    recorder.start()
    for index, raw in enumerate((30_000, 30_050, 30_100)):
        expected = calibration.sample_from_raw(
            raw,
            10.0 + index * 0.01,
            device_t_ms=100.0 + index * 10.0,
        )
        recorder.submit(fs.ForceSample(
            t=expected.t,
            raw=expected.raw,
            hall_v=expected.hall_v,
            displacement_mm=expected.displacement_mm,
            force_n=expected.force_n + 0.1,
            status=expected.status,
            device_t_ms=expected.device_t_ms,
        ))

    with pytest.raises(capture.ForceCaptureError, match="calibration.*force_n"):
        recorder.finish(t_end=10.02)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_verifier_recomputes_force_authenticity_instead_of_trusting_receipt(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0, raw=31_000, force_n=2.75))
    recorder.submit(_sample(10.01, 110.0, raw=31_001, force_n=2.76))
    recorder.finish(t_end=10.01)

    metadata_path = tmp_path / "force_capture.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["signal_authenticity"]["status"] = "FAIL"
    metadata_raw = (
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    metadata_path.write_bytes(metadata_raw)
    marker_path = tmp_path / capture.COMPLETE_NAME
    marker = json.loads(marker_path.read_text())
    marker["force_capture_json_sha256"] = hashlib.sha256(metadata_raw).hexdigest()
    marker_path.write_text(
        json.dumps(
            marker,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )

    with pytest.raises(
        capture.ForceCaptureVerificationError,
        match="authenticity receipt",
    ):
        capture.verify_force_capture(tmp_path)


def test_absent_completion_marker_is_never_a_successful_capture(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.submit(_sample(10.01, 110.0))
    recorder.finish(t_end=10.01)
    (tmp_path / capture.COMPLETE_NAME).unlink()

    with pytest.raises(capture.ForceCaptureVerificationError, match="completion marker"):
        capture.verify_force_capture(tmp_path)


def test_queue_overflow_fails_closed_without_publishing_force_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _recorder(tmp_path, queue_samples=2)
    gate = threading.Event()
    original_run = recorder._run

    def delayed_run() -> None:
        gate.wait(timeout=2.0)
        original_run()

    monkeypatch.setattr(recorder, "_run", delayed_run)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.submit(_sample(10.01, 110.0))
    recorder.submit(_sample(10.02, 120.0))
    gate.set()

    with pytest.raises(capture.ForceCaptureError, match="queue overflow"):
        recorder.finish(t_end=10.02)

    assert not (tmp_path / "force.csv").exists()
    assert not (tmp_path / capture.COMPLETE_NAME).exists()
    state = json.loads((tmp_path / "force_capture.json.inprogress").read_text())
    assert state["status"] == "interrupted"
    assert state["training_eligible"] is False


@pytest.mark.parametrize(
    ("device_times", "message"),
    [
        ([100.0, 110.0, 105.0], "device clock regressed"),
        ([100.0, None, 120.0], "missing device_t_ms"),
        ([100.0, 110.0, 200.0], "device clock gap"),
    ],
)
def test_missing_regressed_or_gapped_device_clock_fails_closed(
    tmp_path: Path, device_times: list[float | None], message: str,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for index, device_t_ms in enumerate(device_times):
        recorder.submit(_sample(10.0 + index * 0.01, device_t_ms))

    with pytest.raises(capture.ForceCaptureError, match=message):
        recorder.finish(t_end=10.02)

    assert not (tmp_path / "force.csv").exists()
    assert not (tmp_path / capture.COMPLETE_NAME).exists()


@pytest.mark.parametrize(
    ("samples", "end", "message"),
    [
        ([(10.00, 100.0), (10.01, 110.0), (10.10, 120.0)], 10.10, "host clock gap"),
        ([(10.00, 100.0), (10.01, 110.0)], 10.10, "coverage gap at End"),
    ],
)
def test_host_gap_or_missing_end_coverage_fails_closed(
    tmp_path: Path,
    samples: list[tuple[float, float]],
    end: float,
    message: str,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    for host_t_s, device_t_ms in samples:
        recorder.submit(_sample(host_t_s, device_t_ms))

    with pytest.raises(capture.ForceCaptureError, match=message):
        recorder.finish(t_end=end)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_serial_reader_error_fails_the_active_capture(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.report_reader_error(RuntimeError("USB serial disconnected"), t=10.005)
    recorder.submit(_sample(10.01, 110.0))

    with pytest.raises(capture.ForceCaptureError, match="serial reader.*disconnected"):
        recorder.finish(t_end=10.01)

    assert not (tmp_path / capture.COMPLETE_NAME).exists()


@pytest.mark.parametrize("unsafe_status", [fs.STATUS_SCREENED_LIMIT, fs.STATUS_HARD_LIMIT])
def test_over_threshold_force_is_recorded_advisory_and_still_seals(
    tmp_path: Path, unsafe_status: str,
) -> None:
    # P0-4: an over-threshold ("unsafe") reading is an advisory physician reference --
    # it is recorded (never dropped) and the capture still seals; it never latches a
    # failure, and the exact over-threshold datum is preserved in force.csv.
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0, status=fs.STATUS_OK))
    recorder.submit(_sample(10.01, 110.0, status=unsafe_status))
    recorder.submit(_sample(10.02, 120.0, status=fs.STATUS_OK))

    summary = recorder.finish(t_end=10.02)

    assert (tmp_path / "force.csv").exists()
    assert (tmp_path / capture.COMPLETE_NAME).exists()
    csv_text = (tmp_path / "force.csv").read_text(encoding="utf-8")
    assert unsafe_status in csv_text
    assert int(summary.n_samples) >= 3


def test_provisional_calibration_cannot_start_a_formal_sidecar(tmp_path: Path) -> None:
    provisional = dict(CALIBRATION, calibrated=False)
    recorder = _recorder(tmp_path, calibration_identity=provisional)

    with pytest.raises(capture.ForceCaptureError, match="calibrated=true"):
        recorder.start()

    assert not (tmp_path / "force.csv.inprogress").exists()
    assert not (tmp_path / capture.COMPLETE_NAME).exists()


def test_abort_preserves_diagnostic_spool_but_never_publishes_completion(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.abort()

    assert (tmp_path / "force.csv.inprogress").is_file()
    assert not (tmp_path / "force.csv").exists()
    assert not (tmp_path / "force_capture.json").exists()
    assert not (tmp_path / capture.COMPLETE_NAME).exists()
    state = json.loads((tmp_path / "force_capture.json.inprogress").read_text())
    assert state["status"] == "interrupted"
    assert state["training_eligible"] is False


def test_invalid_end_still_stops_spool_and_marks_capture_interrupted(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))

    with pytest.raises(ValueError, match="must follow Start"):
        recorder.finish(t_end=10.0)

    assert recorder._thread is not None
    assert not recorder._thread.is_alive()
    assert not (tmp_path / capture.COMPLETE_NAME).exists()
    state = json.loads((tmp_path / "force_capture.json.inprogress").read_text())
    assert state["status"] == "interrupted"


def test_verifier_detects_force_csv_tampering_after_completion(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.submit(_sample(10.01, 110.0))
    recorder.finish(t_end=10.01)
    with (tmp_path / "force.csv").open("a", encoding="utf-8") as handle:
        handle.write("10.02,1,0.1,0.1,0.1,WARN,120\n")

    with pytest.raises(capture.ForceCaptureVerificationError, match="SHA-256"):
        capture.verify_force_capture(tmp_path)


def test_verifier_detects_path_swap_between_hash_and_csv_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.submit(_sample(10.0, 100.0))
    recorder.submit(_sample(10.01, 110.0))
    recorder.finish(t_end=10.01)
    csv_path = tmp_path / "force.csv"
    replacement = csv_path.read_text(encoding="utf-8").replace("2.75", "2.70")
    original_audit = capture._copy_and_validate_csv
    swapped = False

    def swap_path_then_audit(*args, **kwargs):
        nonlocal swapped
        if not swapped and kwargs.get("t0") == pytest.approx(10.0):
            swapped = True
            temporary = tmp_path / "force.replacement.csv"
            temporary.write_text(replacement, encoding="utf-8")
            temporary.replace(csv_path)
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(capture, "_copy_and_validate_csv", swap_path_then_audit)

    with pytest.raises(capture.ForceCaptureVerificationError, match="changed while validating"):
        capture.verify_force_capture(tmp_path)
    assert swapped is True


def test_long_capture_has_fixed_queue_and_bounded_memory(tmp_path: Path) -> None:
    recorder = _recorder(
        tmp_path,
        queue_samples=32,
        checkpoint_samples=1_000,
        max_device_gap_ms=11.0,
        max_host_gap_ms=11.0,
    )
    recorder.start()
    assert recorder.queue_capacity == 32
    assert not hasattr(recorder, "_samples")

    tracemalloc.start()
    count = 10_000
    for index in range(count):
        raw = 30_000 + (index % 201)
        recorder.submit(_sample(
            10.0 + index * 0.01,
            100.0 + index * 10.0,
            raw=raw,
            hall_v=raw / 20_000.0,
            displacement_mm=(raw - 30_000) / 100.0,
            force_n=(raw - 30_000) / 40.0,
        ))
        if index % 16 == 15:
            recorder._queue.join()
    summary = recorder.finish(t_end=10.0 + (count - 1) * 0.01)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert summary.n_samples == count
    assert peak < 8 * 1024 * 1024
    assert recorder.queue_capacity == 32


def test_stream_owns_one_serial_reader_and_native_callback_feeds_sidecar(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = json.loads(
        (FF / "force_feedback_calibration_veo6.json").read_text(encoding="utf-8")
    )
    calibration_payload["calibrated"] = True
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")
    instances = []

    class FakeSensor:
        def __init__(self, port, calibration, **kwargs):
            self.port = port
            self.calibration = calibration
            self.on_sample = kwargs["on_sample"]
            self.on_reader_error = kwargs["on_reader_error"]
            self.starts = 0
            self.stops = 0
            instances.append(self)

        def start(self):
            self.starts += 1
            return self

        def stop(self):
            self.stops += 1

        def read(self):
            return _sample(10.01, 110.0)

    session = tmp_path / "session"
    stream = capture.ForceCaptureStream(
        "/dev/serial/by-id/usb-test-hall",
        calibration_path,
        device_identity=DEVICE,
        sensor_factory=FakeSensor,
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    )
    stream.prepare()
    stream.prepare()
    assert len(instances) == 1
    assert instances[0].starts == 1

    stream.begin_recording(session, t0=10.0)
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        30_000, 10.0, device_t_ms=100.0,
    ))
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        30_050, 10.01, device_t_ms=110.0,
    ))
    result = stream.end_recording(t_end=10.01)
    stream.stop()

    assert result.n_samples == 2
    assert instances[0].stops == 1
    assert capture.verify_force_capture(session).n_samples == 2


def test_two_phase_prearm_keeps_native_sample_arriving_after_camera_t0_before_begin(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = json.loads(
        (FF / "force_feedback_calibration_veo6.json").read_text(encoding="utf-8")
    )
    calibration_payload["calibrated"] = True
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")
    instances = []

    class FakeSensor:
        def __init__(self, _port, _calibration, **kwargs):
            self.calibration = _calibration
            self.on_sample = kwargs["on_sample"]
            instances.append(self)

        def start(self):
            return self

        def stop(self):
            pass

        def read(self):
            return _sample(10.01, 110.0)

    stream = capture.ForceCaptureStream(
        "/dev/serial/by-id/usb-test-hall",
        calibration_path,
        device_identity=DEVICE,
        sensor_factory=FakeSensor,
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    ).prepare()
    session = tmp_path / "two-phase-session"
    stream.prepare_recording(session)
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        29_950, 9.99, device_t_ms=90.0,
    ))

    # Camera Start has now happened at exactly 10.0, but the coordinator has
    # not yet called begin_recording.  Because the force spool was pre-armed,
    # both callback samples remain available for exact boundary filtering.
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        30_000, 10.00, device_t_ms=100.0,
    ))
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        30_050, 10.01, device_t_ms=110.0,
    ))
    stream.begin_recording(t0=10.0)
    instances[0].on_sample(instances[0].calibration.sample_from_raw(
        30_100, 10.02, device_t_ms=120.0,
    ))
    result = stream.end_recording(t_end=10.02)
    stream.stop()

    assert result.n_samples == 3
    assert [float(row["host_t_s"]) for row in _csv_rows(session / "force.csv")] == pytest.approx(
        [10.0, 10.01, 10.02]
    )


def test_stream_reports_reader_fault_to_recorder_before_supervisor(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = json.loads(
        (FF / "force_feedback_calibration_veo6.json").read_text(encoding="utf-8")
    )
    calibration_payload["calibrated"] = True
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")
    events = []

    class FakeSensor:
        def __init__(self, _port, _calibration, **kwargs):
            self.on_sample = kwargs["on_sample"]
            self.on_reader_error = kwargs["on_reader_error"]

        def start(self):
            return self

        def stop(self):
            pass

        def read(self):
            return _sample(10.01, 110.0)

    stream = capture.ForceCaptureStream(
        "/dev/serial/by-id/usb-test-hall",
        calibration_path,
        device_identity=DEVICE,
        sensor_factory=FakeSensor,
        on_capture_fault=lambda _error, _at: events.append("supervisor"),
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    ).prepare()
    session = tmp_path / "fault-session"
    stream.prepare_recording(session)
    stream.begin_recording(t0=10.0)
    recorder = stream._recording
    original = recorder.report_reader_error

    def report(error, *, t):
        events.append("recorder")
        return original(error, t=t)

    recorder.report_reader_error = report
    stream._sensor.on_reader_error(OSError("serial failed"), 10.005)

    assert events == ["recorder", "supervisor"]
    with pytest.raises(capture.ForceCaptureError, match="serial reader failed"):
        stream.end_recording(t_end=10.01)
    stream.stop()


def test_stream_safety_callback_is_transition_only_and_over_threshold_is_recorded_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_payload = json.loads(
        (FF / "force_feedback_calibration_veo6.json").read_text(encoding="utf-8")
    )
    calibration_payload["calibrated"] = True
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")
    calibration = fs.ForceCalibration.from_mapping(calibration_payload)

    def raw_for(status: str) -> int:
        for raw in range(0, calibration.adc_full_scale + 1, 8):
            if calibration.sample_from_raw(raw, 0.0).status == status:
                return raw
        raise AssertionError(f"no raw ADC value found for {status}")

    ok_raw = raw_for(fs.STATUS_OK)
    screened_raw = raw_for(fs.STATUS_SCREENED_LIMIT)
    hard_raw = raw_for(fs.STATUS_HARD_LIMIT)
    gate = threading.Event()

    class FakeSerial:
        def __init__(self):
            self.rows = iter([
                f"90,{ok_raw}\n",       # discarded first parsed line
                f"100,{ok_raw}\n",
                f"110,{screened_raw}\n",
                f"120,{screened_raw}\n",  # same transition: no second callback
                f"130,{hard_raw}\n",
                f"140,{hard_raw}\n",      # same transition: no second callback
                f"150,{ok_raw}\n",        # reset transition state
                f"160,{screened_raw}\n",
            ])

        def reset_input_buffer(self):
            pass

        def readline(self):
            gate.wait(timeout=1.0)
            try:
                return next(self.rows).encode("ascii")
            except StopIteration:
                time.sleep(0.002)
                return b""

        def close(self):
            pass

    serial_module = type(sys)("serial")
    serial_module.Serial = lambda *_args, **_kwargs: FakeSerial()
    monkeypatch.setitem(sys.modules, "serial", serial_module)
    safety_events: list[str] = []

    def safety_callback(sample: fs.ForceSample) -> None:
        # Notification only -- force is advisory, so the callback never controls capture.
        safety_events.append(sample.status)

    stream = capture.ForceCaptureStream(
        "/dev/serial/by-id/usb-test-hall",
        calibration_path,
        device_identity=DEVICE,
        on_safety_event=safety_callback,
        max_device_gap_ms=25.0,
        max_host_gap_ms=25.0,
    ).prepare()
    session = tmp_path / "safety-session"
    stream.prepare_recording(session)
    start = time.monotonic()
    stream.begin_recording(t0=start)
    gate.set()
    deadline = time.monotonic() + 0.5
    while len(safety_events) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)

    # P0-4: an over-threshold ("unsafe") reading is an advisory physician reference --
    # the recorder keeps recording, the reading is preserved, and the capture COMPLETES.
    summary = stream.end_recording(t_end=time.monotonic())
    stream.stop()

    assert safety_events == [
        fs.STATUS_SCREENED_LIMIT,
        fs.STATUS_HARD_LIMIT,
        fs.STATUS_SCREENED_LIMIT,
    ]
    assert (session / capture.COMPLETE_NAME).exists()
    # the over-threshold samples the physician needs are preserved in the CSV, not dropped
    csv_text = (session / capture.CSV_NAME).read_text(encoding="utf-8")
    assert str(hard_raw) in csv_text
    assert int(summary.n_samples) >= 5


def test_force_sensor_reports_each_native_sample_and_serial_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = fs.ForceCalibration.from_json(FF / "force_feedback_calibration_veo6.json")
    samples: list[fs.ForceSample] = []
    errors: list[tuple[BaseException, float]] = []

    class FakeSerial:
        def __init__(self):
            self.calls = 0

        def reset_input_buffer(self):
            pass

        def readline(self):
            self.calls += 1
            if self.calls == 1:
                return b"90,30000\n"  # first parsed line is intentionally discarded
            if self.calls == 2:
                return b"100,31000\n"
            if self.calls == 3:
                raise OSError("serial read failed")
            time.sleep(0.002)
            return b""

        def close(self):
            pass

    serial_module = type(sys)("serial")
    serial_module.Serial = lambda *_args, **_kwargs: FakeSerial()
    monkeypatch.setitem(sys.modules, "serial", serial_module)
    sensor = fs.ForceSensor(
        "/dev/fake",
        calibration,
        on_sample=samples.append,
        on_reader_error=lambda exc, at: errors.append((exc, at)),
    )
    sensor.start()
    deadline = time.monotonic() + 1.0
    while (not samples or not errors) and time.monotonic() < deadline:
        time.sleep(0.005)
    sensor.stop()

    assert len(samples) == 1
    assert samples[0].device_t_ms == pytest.approx(100.0)
    assert len(errors) == 1
    assert isinstance(errors[0][0], OSError)
    assert errors[0][1] > 0.0
