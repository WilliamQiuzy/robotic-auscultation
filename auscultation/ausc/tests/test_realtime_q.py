from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

AUSC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUSC))

import realtime_q  # noqa: E402


MODEL_SHA256 = "a" * 64
MODEL_RELEASE_ID = f"clinical-q-release-test-{MODEL_SHA256[:20]}"
CALIBRATION_SHA256 = "c" * 64
CALIBRATION_RELEASE_ID = (
    f"clinical-q-calibration-test-{CALIBRATION_SHA256[:20]}"
)


def test_missing_or_unsafe_force_never_fabricates_open_gate():
    assert realtime_q.safe_force_value(None) is None
    assert realtime_q.safe_force_value(SimpleNamespace(force_n=4.0, status="NO_DATA")) is None
    assert realtime_q.safe_force_value(SimpleNamespace(force_n=4.0, status="STALE")) is None
    assert realtime_q.safe_force_value(SimpleNamespace(force_n=4.0, status="OK")) == 4.0


def test_realtime_calibration_uses_environment_then_persistent_default(monkeypatch, tmp_path):
    selected = tmp_path / "commissioned.json"
    selected.write_text("{}")
    monkeypatch.setenv("WAM_HALL_CALIBRATION", str(selected))
    assert realtime_q.resolve_hall_calibration(None) == selected

    monkeypatch.delenv("WAM_HALL_CALIBRATION")
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent = tmp_path / "wam_data/config/calibration/force_feedback_calibration_veo6_tared.json"
    persistent.parent.mkdir(parents=True)
    persistent.write_text("{}")
    assert realtime_q.resolve_hall_calibration(None) == persistent


def test_realtime_calibration_fails_closed_when_selected_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hall calibration"):
        realtime_q.resolve_hall_calibration(tmp_path / "missing.json")


def _quality_policy() -> dict:
    return {
        "schema_version": "wam_quality_policy_v1",
        "policy_id": "clinical-q-test-v1",
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
                "release_id": CALIBRATION_RELEASE_ID,
                "sha256": CALIBRATION_SHA256,
            },
            "transform_to_p_usable": "identity",
            "output_range": [0.0, 1.0],
        },
    }


def _audio_window(value: float, *, t_start: float):
    samples = np.linspace(value, value + 0.01, 4_000, dtype=np.float32)
    source_start = int(round(t_start * 48_000))
    return SimpleNamespace(
        samples=samples,
        sr=16_000,
        t_start=t_start,
        t_end=t_start + 0.25,
        status="OK",
        raw_source={
            "source_channel": 1,
            "source_sample_start": source_start,
            "source_sample_end": source_start + 12_000,
            "source_sample_rate": 48_000,
            "resampler_identity": {
                "algorithm": "polyphase_fir_v1",
                "input_sample_rate": 48_000,
                "output_sample_rate": 16_000,
            },
        },
    )


def _force_sample(
    *,
    t: float,
    status: str = "OK",
    raw: int = 30_000,
    hall_v: float = 1.75,
    displacement_mm: float = 2.5,
    force_n: float = 4.0,
    device_t_ms: float | None = None,
):
    return SimpleNamespace(
        t=t,
        raw=raw,
        hall_v=hall_v,
        displacement_mm=displacement_mm,
        force_n=force_n,
        status=status,
        device_t_ms=(t * 1_000.0 if device_t_ms is None else device_t_ms),
    )


def _production_quality(
    session_dir: Path,
    scorer=lambda _samples, _sr: 0.0,
    *,
    packet_log_writer=None,
):
    if packet_log_writer is None:
        packet_log_writer = realtime_q.QualityPacketLogWriter(session_dir)
    return realtime_q.StreamingQuality(
        scorer=scorer,
        mode="production",
        model_release_id=MODEL_RELEASE_ID,
        model_sha256=MODEL_SHA256,
        policy=_quality_policy(),
        packet_log_writer=packet_log_writer,
    )


FORCE_CSV_FIELDS = (
    "host_t_s",
    "raw",
    "hall_v",
    "displacement_mm",
    "force_n",
    "status",
    "device_t_ms",
)


def _force_csv_row(force: dict[str, object]) -> str:
    values: list[str] = []
    for field in FORCE_CSV_FIELDS:
        value = force[field]
        if field == "raw":
            values.append(str(value))
        elif field == "status":
            values.append(str(value))
        else:
            values.append(format(float(value), ".17g"))
    return ",".join(values)


def _write_force_csv(
    session_dir: Path,
    packets: list[dict[str, object]],
    *,
    extra_rows: list[dict[str, object]] | None = None,
) -> Path:
    rows = [packet["force_sample"] for packet in packets]
    if extra_rows:
        rows.extend(extra_rows)
    rows.sort(key=lambda row: float(row["host_t_s"]))
    path = session_dir / "force.csv"
    path.write_text(
        ",".join(FORCE_CSV_FIELDS)
        + "\n"
        + "".join(_force_csv_row(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _finalized_quality_session(
    session_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    writer = realtime_q.QualityPacketLogWriter(session_dir)
    quality = _production_quality(
        session_dir,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    packets = []
    for sequence, value, timestamp in (
        (100, 0.1, 20.25),
        (101, 0.2, 20.55),
        (102, 0.3, 20.85),
    ):
        packets.append(quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(
                t=timestamp - 0.05,
                raw=30_000 + sequence,
                hall_v=1.7 + sequence / 100_000.0,
                displacement_mm=2.0 + sequence / 10_000.0,
                device_t_ms=timestamp * 1_000.0,
            ),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )["packet"])
    _write_force_csv(session_dir, packets)
    receipt = quality.quality_authenticity(now_monotonic_s=20.9)
    return packets, writer.finalize(receipt, now_monotonic_s=20.9)


def _rehash_packet_chain(packets: list[dict[str, object]]) -> None:
    previous = None
    for packet in packets:
        packet["previous_packet_chain_sha256"] = previous
        packet_sha256, chain_sha256 = realtime_q._quality_packet_hashes(packet)
        packet["packet_sha256"] = packet_sha256
        packet["packet_chain_sha256"] = chain_sha256
        previous = chain_sha256


def _bind_receipt_to_packets(
    receipt: dict[str, object],
    packets: list[dict[str, object]],
) -> dict[str, object]:
    candidate = realtime_q._json_clone(receipt)
    last = realtime_q._json_clone(packets[-1])
    candidate["last_packet"] = last
    candidate["last_packet_sha256"] = last["packet_sha256"]
    candidate["packet_chain_sha256"] = last["packet_chain_sha256"]
    candidate["last_audio_content_sha256"] = last["audio_window"][
        "content_sha256"
    ]
    return candidate


def test_default_production_quality_is_unready_and_never_fabricates_zero():
    quality = realtime_q.StreamingQuality()

    readiness = quality.readiness()
    assert readiness["ready"] is False
    assert readiness["status"] == "NOT_READY"
    assert "pinned production Q scorer" in readiness["blockers"]
    with pytest.raises(realtime_q.QualityNotReadyError, match="not ready"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=1.0),
            force_sample=_force_sample(t=1.2),
            inference_sequence=1,
            inference_monotonic_s=1.25,
            observed_monotonic_s=1.26,
        )


def test_non_callable_model_binding_is_not_production_ready():
    quality = realtime_q.StreamingQuality(
        scorer=object(),
        mode="production",
        model_release_id=MODEL_RELEASE_ID,
        model_sha256=MODEL_SHA256,
        policy=_quality_policy(),
    )

    assert quality.readiness()["production_ready"] is False
    with pytest.raises(realtime_q.QualityNotReadyError, match="callable"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=1.0),
            force_sample=_force_sample(t=1.2),
            inference_sequence=1,
            inference_monotonic_s=1.25,
            observed_monotonic_s=1.26,
        )


def test_model_release_id_must_bind_its_model_sha256_suffix():
    with pytest.raises(ValueError, match="content-addressed"):
        realtime_q.StreamingQuality(
            scorer=lambda _samples, _sr: 0.5,
            mode="production",
            model_release_id="clinical-q-release-test-" + "b" * 20,
            model_sha256=MODEL_SHA256,
            policy=_quality_policy(),
        )


def test_calibration_release_id_must_bind_its_sha256_suffix():
    policy = _quality_policy()
    policy["output_contract"]["calibration"]["release_id"] = (
        "clinical-q-calibration-test-" + "b" * 20
    )

    with pytest.raises(ValueError, match="content-addressed"):
        realtime_q.StreamingQuality(
            scorer=lambda _samples, _sr: 0.5,
            mode="production",
            model_release_id=MODEL_RELEASE_ID,
            model_sha256=MODEL_SHA256,
            policy=policy,
        )


def test_band_energy_proxy_is_rejected_as_a_production_scorer():
    with pytest.raises(ValueError, match="diagnostic-only"):
        realtime_q.StreamingQuality(
            scorer=realtime_q.band_energy_proxy,
            mode="production",
            model_release_id="forbidden-proxy-release",
            model_sha256="b" * 64,
            policy=_quality_policy(),
        )


def test_diagnostic_proxy_is_explicit_and_cannot_issue_production_receipt():
    quality = realtime_q.StreamingQuality(
        scorer=realtime_q.band_energy_proxy,
        mode="diagnostic",
        policy=realtime_q.diagnostic_quality_policy(
            window_s=0.25,
            force_band=(3.0, 5.0),
        ),
    )

    result = quality.update(
        audio_window=_audio_window(0.1, t_start=1.0),
        force_sample=_force_sample(t=1.2),
        inference_sequence=1,
        inference_monotonic_s=1.25,
        observed_monotonic_s=1.26,
    )

    assert result["packet"]["scorer_scope"] == "diagnostic_only"
    assert result["packet"]["production_eligible"] is False
    assert "model_input" not in result["packet"]["audio_window"]
    assert quality.readiness()["production_ready"] is False
    with pytest.raises(realtime_q.QualityNotReadyError, match="diagnostic-only"):
        quality.quality_authenticity(now_monotonic_s=1.3)


def test_live_demo_runtime_is_explicitly_diagnostic_only():
    quality = realtime_q.build_diagnostic_quality_runtime(
        window_s=8.0,
        force_band=(3.0, 5.0),
    )

    readiness = quality.readiness()
    assert readiness["status"] == "READY"
    assert readiness["mode"] == "diagnostic"
    assert readiness["production_ready"] is False
    assert quality.scorer is realtime_q.band_energy_proxy
    assert quality.policy["policy_id"] == "diagnostic-band-energy-proxy-v1"


def test_constant_zero_model_output_is_live_when_packet_evidence_advances(tmp_path):
    quality = _production_quality(tmp_path, scorer=lambda _samples, _sr: 0.0)
    first_window = _audio_window(0.1, t_start=1.0)
    second_window = _audio_window(0.2, t_start=1.3)

    first = quality.update(
        audio_window=first_window,
        force_sample=_force_sample(t=1.2),
        inference_sequence=41,
        inference_monotonic_s=1.25,
        observed_monotonic_s=1.26,
    )
    second = quality.update(
        audio_window=second_window,
        force_sample=_force_sample(t=1.5, status="WARN"),
        inference_sequence=42,
        inference_monotonic_s=1.55,
        observed_monotonic_s=1.56,
    )
    receipt = quality.quality_authenticity(now_monotonic_s=1.6)

    expected_audio_sha = hashlib.sha256(
        np.asarray(second_window.samples, dtype="<f4").tobytes()
    ).hexdigest()
    packet = second["packet"]
    assert first["q_raw"] == second["q_raw"] == 0.0
    assert packet["model"] == {
        "release_id": MODEL_RELEASE_ID,
        "model_sha256": MODEL_SHA256,
        "policy_id": "clinical-q-test-v1",
        "policy_sha256": receipt["policy_sha256"],
        "output_contract": _quality_policy()["output_contract"],
    }
    assert packet["inference_sequence"] == 42
    assert packet["inference_monotonic_s"] == pytest.approx(1.55)
    assert packet["observed_monotonic_s"] == pytest.approx(1.56)
    assert {
        key: value
        for key, value in packet["audio_window"].items()
        if key not in ("model_input", "raw_source")
    } == {
        "t_start_monotonic_s": pytest.approx(1.3),
        "t_end_monotonic_s": pytest.approx(1.55),
        "sample_rate": 16_000,
        "sample_count": 4_000,
        "content_sha256": expected_audio_sha,
        "encoding": "float32_le_mono",
        "status": "OK",
    }
    assert packet["audio_window"]["model_input"] == {
        "schema_version": "wam_quality_model_input_v1",
        "path": "quality_model_inputs/input_00000000000000000042.f32le",
        "inference_sequence": 42,
        "dtype": "<f4",
        "shape": [4_000],
        "bytes": 16_000,
        "sha256": expected_audio_sha,
    }
    raw_source = packet["audio_window"]["raw_source"]
    assert raw_source["schema_version"] == "wam_quality_audio_raw_source_v1"
    assert raw_source["source_channel"] == 1
    assert raw_source["source_sample_end"] > raw_source["source_sample_start"]
    assert raw_source["resampler_identity"] == {
        "algorithm": "polyphase_fir_v1",
        "input_sample_rate": 48_000,
        "output_sample_rate": 16_000,
    }
    assert packet["force_sample"] == {
        "host_t_s": pytest.approx(1.5),
        "raw": 30_000,
        "hall_v": pytest.approx(1.75),
        "displacement_mm": pytest.approx(2.5),
        "force_n": pytest.approx(4.0),
        "status": "WARN",
        "device_t_ms": pytest.approx(1_500.0),
    }
    assert receipt["schema_version"] == "wam_quality_authenticity_v2"
    assert receipt["policy_version"] == "quality_packet_liveness_v3"
    assert receipt["status"] == "PASS"
    assert receipt["promotion_eligible"] is False
    assert receipt["packet_count"] == 2
    assert receipt["first_inference_sequence"] == 41
    assert receipt["last_inference_sequence"] == 42
    assert receipt["checks"] == {
        "audio_input_progress": "PASS",
        "lossless_model_input_binding": "PASS",
        "force_raw_evidence": "PASS",
        "inference_freshness": "PASS",
        "inference_sequence_progress": "PASS",
        "inference_timestamp_progress": "PASS",
        "model_identity": "PASS",
        "output_semantics": "PASS",
        "packet_hash_chain": "PASS",
    }


def test_poor_probability_is_explicitly_transformed_to_p_usable(tmp_path):
    policy = _quality_policy()
    policy["output_contract"]["positive_class"] = "poor"
    policy["output_contract"]["transform_to_p_usable"] = "one_minus"
    poor_sha256 = "d" * 64
    quality = realtime_q.StreamingQuality(
        scorer=lambda _samples, _sr: 0.8,
        mode="production",
        model_release_id=f"clinical-q-poor-release-test-{poor_sha256[:20]}",
        model_sha256=poor_sha256,
        policy=policy,
        packet_log_writer=realtime_q.QualityPacketLogWriter(tmp_path),
    )

    result = quality.update(
        audio_window=_audio_window(0.1, t_start=1.0),
        force_sample=_force_sample(t=1.2),
        inference_sequence=1,
        inference_monotonic_s=1.25,
        observed_monotonic_s=1.26,
    )

    assert result["q_raw"] == pytest.approx(0.8)
    assert result["p_usable_raw"] == pytest.approx(0.2)
    assert result["p_usable"] == pytest.approx(0.1)
    assert result["packet"]["model"]["output_contract"] == policy["output_contract"]


def test_output_contract_rejects_a_direction_transform_mismatch():
    policy = _quality_policy()
    policy["output_contract"]["positive_class"] = "poor"

    with pytest.raises(ValueError, match="transform_to_p_usable"):
        poor_sha256 = "d" * 64
        realtime_q.StreamingQuality(
            scorer=lambda _samples, _sr: 0.8,
            mode="production",
            model_release_id=f"clinical-q-poor-release-test-{poor_sha256[:20]}",
            model_sha256=poor_sha256,
            policy=policy,
        )


@pytest.mark.parametrize(
    ("sequence", "timestamp", "audio_value", "message"),
    [
        (7, 2.30, 0.2, "sequence"),
        (8, 2.25, 0.2, "timestamp"),
        (8, 2.30, 0.1, "input hash"),
    ],
)
def test_quality_liveness_rejects_frozen_counter_timestamp_or_input(
    sequence, timestamp, audio_value, message, tmp_path,
):
    quality = _production_quality(tmp_path)
    quality.update(
        audio_window=_audio_window(0.1, t_start=2.0),
        force_sample=_force_sample(t=2.2),
        inference_sequence=7,
        inference_monotonic_s=2.25,
        observed_monotonic_s=2.26,
    )

    with pytest.raises(realtime_q.QualityPacketError, match=message):
        quality.update(
            audio_window=_audio_window(audio_value, t_start=2.05),
            force_sample=_force_sample(t=2.25),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=2.31,
        )


def test_frozen_audio_is_rejected_before_the_pinned_model_runs_again(tmp_path):
    calls = []

    def scorer(_samples, _sr):
        calls.append("called")
        return 0.5

    quality = _production_quality(tmp_path, scorer=scorer)
    quality.update(
        audio_window=_audio_window(0.1, t_start=2.0),
        force_sample=_force_sample(t=2.2),
        inference_sequence=7,
        inference_monotonic_s=2.25,
        observed_monotonic_s=2.26,
    )

    with pytest.raises(realtime_q.QualityPacketError, match="input hash"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=2.1),
            force_sample=_force_sample(t=2.3),
            inference_sequence=8,
            inference_monotonic_s=2.35,
            observed_monotonic_s=2.36,
        )

    assert calls == ["called"]


def test_stale_audio_window_is_rejected_before_the_pinned_model_runs(tmp_path):
    calls = []

    def scorer(_samples, _sr):
        calls.append("called")
        return 0.5

    quality = _production_quality(tmp_path, scorer=scorer)

    with pytest.raises(realtime_q.QualityPacketError, match="audio window is stale"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=1.0),
            force_sample=_force_sample(t=1.95),
            inference_sequence=1,
            inference_monotonic_s=2.0,
            observed_monotonic_s=2.01,
        )

    assert calls == []


def test_unknown_force_status_is_rejected_before_the_pinned_model_runs(tmp_path):
    calls = []

    def scorer(_samples, _sr):
        calls.append("called")
        return 0.5

    quality = _production_quality(tmp_path, scorer=scorer)

    with pytest.raises(realtime_q.QualityPacketError, match="force status"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=1.0),
            force_sample=_force_sample(t=1.2, status="UNKNOWN"),
            inference_sequence=1,
            inference_monotonic_s=1.25,
            observed_monotonic_s=1.26,
        )

    assert calls == []


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1, "0.5"])
def test_invalid_pinned_model_output_fails_closed(value, tmp_path):
    quality = _production_quality(
        tmp_path, scorer=lambda _samples, _sr: value,
    )

    with pytest.raises(realtime_q.QualityPacketError, match="scorer output"):
        quality.update(
            audio_window=_audio_window(0.1, t_start=1.0),
            force_sample=_force_sample(t=1.2),
            inference_sequence=1,
            inference_monotonic_s=1.25,
            observed_monotonic_s=1.26,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("ema", "0.5"), ("max_audio_age_s", True)],
)
def test_quality_policy_rejects_non_numeric_json_values(field, value):
    policy = _quality_policy()
    policy[field] = value

    with pytest.raises(ValueError, match=field):
        realtime_q.StreamingQuality(
            scorer=lambda _samples, _sr: 0.5,
            model_release_id=MODEL_RELEASE_ID,
            model_sha256=MODEL_SHA256,
            policy=policy,
        )


def test_quality_receipt_requires_a_fresh_inference_packet(tmp_path):
    quality = _production_quality(tmp_path)
    for sequence, value, timestamp in ((1, 0.1, 3.25), (2, 0.2, 3.55)):
        quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )

    with pytest.raises(realtime_q.QualityPacketError, match="stale"):
        quality.quality_authenticity(now_monotonic_s=4.1)


def test_quality_receipt_keeps_an_immutable_last_packet_binding(tmp_path):
    quality = _production_quality(
        tmp_path, scorer=lambda _samples, _sr: 0.5,
    )
    for sequence, value, timestamp in ((1, 0.1, 3.25), (2, 0.2, 3.55)):
        result = quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05, status="WARN"),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )

    result["packet"]["force_sample"]["status"] = "TAMPERED"
    receipt = quality.quality_authenticity(now_monotonic_s=3.6)
    assert receipt["last_packet"]["force_sample"]["status"] == "WARN"
    assert receipt["last_packet"]["packet_sha256"] == receipt["last_packet_sha256"]

    receipt["policy"]["force_band_n"][0] = -1.0
    fresh_receipt = quality.quality_authenticity(now_monotonic_s=3.6)
    assert fresh_receipt["policy"]["force_band_n"] == [3.0, 5.0]


def test_independent_verifier_recomputes_last_packet_and_chain_hashes(tmp_path):
    quality = _production_quality(
        tmp_path, scorer=lambda _samples, _sr: 0.5,
    )
    for sequence, value, timestamp in ((1, 0.1, 3.25), (2, 0.2, 3.55)):
        quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )
    receipt = quality.quality_authenticity(now_monotonic_s=3.6)

    verified = realtime_q.verify_quality_authenticity(
        receipt,
        now_monotonic_s=3.6,
    )
    assert verified["status"] == "PASS"
    assert receipt["last_packet"]["hash_contract"] == "wam_quality_packet_hash_v2"
    assert receipt["last_packet"]["packet_chain_sha256"] == receipt["packet_chain_sha256"]

    packet_tamper = realtime_q._json_clone(receipt)
    packet_tamper["last_packet"]["result"]["q_raw"] = 0.6
    with pytest.raises(realtime_q.QualityPacketError, match="packet SHA-256"):
        realtime_q.verify_quality_authenticity(
            packet_tamper,
            now_monotonic_s=3.6,
        )

    chain_tamper = realtime_q._json_clone(receipt)
    chain_tamper["last_packet"]["packet_chain_sha256"] = "f" * 64
    with pytest.raises(realtime_q.QualityPacketError, match="chain SHA-256"):
        realtime_q.verify_quality_authenticity(
            chain_tamper,
            now_monotonic_s=3.6,
        )

    count_tamper = realtime_q._json_clone(receipt)
    count_tamper["packet_count"] = 2.5
    with pytest.raises(realtime_q.QualityPacketError, match="progress evidence"):
        realtime_q.verify_quality_authenticity(
            count_tamper,
            now_monotonic_s=3.6,
        )


def test_offline_verifier_rejects_non_content_addressed_model_release(tmp_path):
    _, log_receipt = _finalized_quality_session(tmp_path)
    receipt = realtime_q._json_clone(log_receipt["quality_authenticity"])
    receipt["model"]["release_id"] = "forged-quality-model"
    receipt["last_packet"]["model"]["release_id"] = "forged-quality-model"
    packet_sha256, chain_sha256 = realtime_q._quality_packet_hashes(
        receipt["last_packet"]
    )
    receipt["last_packet"]["packet_sha256"] = packet_sha256
    receipt["last_packet"]["packet_chain_sha256"] = chain_sha256
    receipt["last_packet_sha256"] = packet_sha256
    receipt["packet_chain_sha256"] = chain_sha256

    with pytest.raises(realtime_q.QualityPacketError, match="content-addressed"):
        realtime_q.verify_quality_authenticity(
            receipt,
            now_monotonic_s=20.9,
        )

def test_full_chain_verifier_requires_every_packet_and_stays_non_promotable(tmp_path):
    quality = _production_quality(
        tmp_path,
        scorer=lambda samples, _sr: float(np.mean(samples)),
    )
    packets = []
    for sequence, value, timestamp in (
        (10, 0.1, 3.25),
        (11, 0.2, 3.55),
        (12, 0.3, 3.85),
    ):
        result = quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )
        packets.append(realtime_q._json_clone(result["packet"]))
    receipt = quality.quality_authenticity(now_monotonic_s=3.9)
    force_csv = _write_force_csv(tmp_path, packets)

    verified = realtime_q.verify_quality_packet_chain(
        packets,
        receipt,
        now_monotonic_s=3.9,
        model_input_dir=quality.packet_log_writer.input_staging_path,
        force_csv_path=force_csv,
    )
    assert verified["schema_version"] == "wam_quality_packet_chain_verification_v2"
    assert verified["status"] == "PASS"
    assert verified["packet_count"] == 3
    assert len(verified["packet_log_sha256"]) == 64
    assert verified["inference_liveness"]["status"] == "PASS"
    assert verified["output_nondegeneracy"]["status"] == "PASS"
    assert verified["promotion_eligible"] is False
    assert receipt["promotion_eligible"] is False

    with pytest.raises(realtime_q.QualityPacketError, match="chain|sequence|count"):
        realtime_q.verify_quality_packet_chain(
            [packets[0], packets[2]],
            receipt,
            now_monotonic_s=3.9,
            model_input_dir=quality.packet_log_writer.input_staging_path,
            force_csv_path=force_csv,
        )


@pytest.mark.parametrize(
    "scorer",
    [
        lambda _samples, _sr: 0.5,
        lambda samples, _sr: 0.5 + float(np.mean(samples)) * 1e-8,
    ],
    ids=["exact-constant", "near-zero-range"],
)
def test_full_chain_separates_live_inference_from_degenerate_output(
    scorer, tmp_path,
):
    quality = _production_quality(tmp_path, scorer=scorer)
    packets = []
    for sequence, value, timestamp in (
        (20, 0.1, 5.25),
        (21, 0.2, 5.55),
        (22, 0.3, 5.85),
    ):
        result = quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )
        packets.append(realtime_q._json_clone(result["packet"]))
    receipt = quality.quality_authenticity(now_monotonic_s=5.9)
    force_csv = _write_force_csv(tmp_path, packets)

    verified = realtime_q.verify_quality_packet_chain(
        packets,
        receipt,
        now_monotonic_s=5.9,
        model_input_dir=quality.packet_log_writer.input_staging_path,
        force_csv_path=force_csv,
    )

    assert verified["chain_integrity"] == "PASS"
    assert verified["inference_liveness"]["status"] == "PASS"
    nondegeneracy = verified["output_nondegeneracy"]
    assert nondegeneracy["policy_version"] == "quality_output_nondegeneracy_v2"
    assert nondegeneracy["status"] == "FAIL"
    assert nondegeneracy["minimum_range"] == pytest.approx(1e-4)
    assert nondegeneracy["q_raw_range"] <= nondegeneracy["minimum_range"]
    assert nondegeneracy["p_usable_raw_range"] <= nondegeneracy["minimum_range"]
    assert verified["status"] == "FAIL"
    assert verified["training_signal_eligible"] is False
    assert verified["promotion_eligible"] is False


def test_full_chain_recomputes_ema_instead_of_trusting_packet_value(tmp_path):
    packets, log_receipt = _finalized_quality_session(tmp_path)
    tampered = realtime_q._json_clone(packets)
    tampered[1]["result"]["p_usable"] = 0.99
    _rehash_packet_chain(tampered)
    receipt = _bind_receipt_to_packets(
        log_receipt["quality_authenticity"], tampered
    )

    with pytest.raises(realtime_q.QualityPacketError, match="EMA"):
        realtime_q.verify_quality_packet_chain(
            tampered,
            receipt,
            now_monotonic_s=20.9,
            model_input_dir=tmp_path / "quality_model_inputs",
            force_csv_path=tmp_path / "force.csv",
        )


def test_full_chain_requires_at_least_three_packets_for_nondegeneracy(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = _production_quality(
        tmp_path,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    packets = []
    for sequence, value, timestamp in (
        (1, 0.1, 3.25),
        (2, 0.3, 3.55),
    ):
        packets.append(quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )["packet"])
    receipt = quality.quality_authenticity(now_monotonic_s=3.6)
    force_csv = _write_force_csv(tmp_path, packets)

    verified = realtime_q.verify_quality_packet_chain(
        packets,
        receipt,
        now_monotonic_s=3.6,
        model_input_dir=writer.input_staging_path,
        force_csv_path=force_csv,
    )

    assert verified["status"] == "FAIL"
    assert verified["output_nondegeneracy"]["minimum_packet_count"] == 3
    assert verified["output_nondegeneracy"]["packet_count"] == 2


@pytest.mark.parametrize(
    "samples",
    [
        np.array([0.0, np.nan], dtype="<f4"),
        np.zeros(64, dtype="<f4"),
        np.full(64, 1e-9, dtype="<f4"),
        (
            np.tile(np.asarray([-1.0, 0.0, 1.0], dtype="<f4"), 64)
            / 32_767.0
        ),
    ],
    ids=["non-finite", "silent", "near-zero", "lsb-scale-noise"],
)
def test_quality_writer_rejects_nonfinite_or_degenerate_model_input(
    tmp_path, samples,
):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)

    with pytest.raises(realtime_q.QualityPacketError, match="finite|signal"):
        writer.persist_model_input(
            inference_sequence=0,
            raw=samples.tobytes(order="C"),
        )


def test_offline_inventory_rejects_rehashed_silent_model_input(tmp_path):
    packets, log_receipt = _finalized_quality_session(tmp_path)
    tampered = realtime_q._json_clone(packets)
    middle = tampered[1]
    input_path = tmp_path / middle["audio_window"]["model_input"]["path"]
    silent = np.zeros(
        middle["audio_window"]["sample_count"], dtype="<f4"
    ).tobytes(order="C")
    input_path.write_bytes(silent)
    silent_sha256 = hashlib.sha256(silent).hexdigest()
    middle["audio_window"]["content_sha256"] = silent_sha256
    middle["audio_window"]["model_input"]["sha256"] = silent_sha256
    _rehash_packet_chain(tampered)
    receipt = _bind_receipt_to_packets(
        log_receipt["quality_authenticity"], tampered
    )

    with pytest.raises(realtime_q.QualityPacketError, match="signal"):
        realtime_q.verify_quality_packet_chain(
            tampered,
            receipt,
            now_monotonic_s=20.9,
            model_input_dir=tmp_path / "quality_model_inputs",
            force_csv_path=tmp_path / "force.csv",
        )


def test_quality_packet_writer_durably_publishes_a_verified_full_log(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    consumed: list[bytes] = []

    def scorer(samples, _sr):
        consumed.append(samples.tobytes(order="C"))
        return float(np.mean(samples))

    quality = _production_quality(
        tmp_path,
        scorer=scorer,
        packet_log_writer=writer,
    )
    packets = []
    for sequence, value, timestamp in (
        (30, 0.1, 7.25),
        (31, 0.2, 7.55),
        (32, 0.3, 7.85),
    ):
        packets.append(quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(
                t=timestamp - 0.05,
                raw=30_000 + sequence,
                hall_v=1.7 + sequence / 100_000.0,
                displacement_mm=2.0 + sequence / 10_000.0,
                device_t_ms=timestamp * 1_000.0,
            ),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )["packet"])
    receipt = quality.quality_authenticity(now_monotonic_s=7.9)
    _write_force_csv(tmp_path, packets)

    log_receipt = writer.finalize(receipt, now_monotonic_s=7.9)

    final_path = tmp_path / "quality_inference_packets.jsonl"
    receipt_path = tmp_path / "quality_inference_authenticity.json"
    input_root = tmp_path / "quality_model_inputs"
    assert final_path.is_file()
    assert receipt_path.is_file()
    assert input_root.is_dir() and not input_root.is_symlink()
    assert not (tmp_path / ".quality_inference_packets.jsonl.inprogress").exists()
    assert not (tmp_path / ".quality_model_inputs.inprogress").exists()
    assert log_receipt["schema_version"] == "wam_quality_packet_log_authenticity_v3"
    assert log_receipt["status"] == "PASS"
    assert log_receipt["packet_count"] == 3
    assert log_receipt["sha256"] == hashlib.sha256(final_path.read_bytes()).hexdigest()
    assert log_receipt["promotion_eligible"] is False
    inventory = log_receipt["model_input_inventory"]
    assert inventory["status"] == "PASS"
    assert inventory["file_count"] == 3
    assert inventory["total_bytes"] == 3 * 4_000 * 4
    assert len(inventory["files"]) == 3
    for index, (sequence, packet, consumed_bytes) in enumerate(
        zip((30, 31, 32), packets, consumed)
    ):
        claim = packet["audio_window"]["model_input"]
        expected_path = (
            "quality_model_inputs/"
            f"input_{sequence:020d}.f32le"
        )
        assert claim == inventory["files"][index]
        assert claim["path"] == expected_path
        assert claim["dtype"] == "<f4"
        assert claim["shape"] == [4_000]
        assert claim["bytes"] == 4_000 * 4
        persisted = (tmp_path / expected_path).read_bytes()
        assert persisted == consumed_bytes
        assert claim["sha256"] == hashlib.sha256(consumed_bytes).hexdigest()
        assert packet["audio_window"]["content_sha256"] == claim["sha256"]
    force_identity = log_receipt["force_input_identity"]
    assert force_identity["schema_version"] == "wam_quality_force_input_identity_v2"
    assert force_identity["status"] == "PASS"
    assert force_identity["path"] == "force.csv"
    assert force_identity["force_csv_row_count"] == 3
    assert force_identity["sample_count"] == 3
    assert force_identity["force_csv_sha256"] == hashlib.sha256(
        (tmp_path / "force.csv").read_bytes()
    ).hexdigest()
    assert [match["inference_sequence"] for match in force_identity["matches"]] == [
        30, 31, 32,
    ]
    assert log_receipt["chain_verification"]["force_input_identity"] == (
        force_identity
    )
    assert log_receipt["promotion_blockers"] == [
        "quality packet log receipt alone is not bound by the final multi-channel "
        "training finalization"
    ]
    assert json.loads(receipt_path.read_text()) == log_receipt
    assert realtime_q.verify_quality_packet_log(tmp_path) == log_receipt
    assert len(realtime_q.read_quality_packet_log(final_path)) == 3

    tampered_receipt = json.loads(receipt_path.read_text())
    tampered_receipt["packet_count"] += 1
    receipt_path.write_text(
        json.dumps(tampered_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(realtime_q.QualityPacketError, match="full-chain|count"):
        realtime_q.verify_quality_packet_log(tmp_path)


def test_quality_packet_writer_rejects_a_missing_middle_packet(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    source = _production_quality(
        source_dir, scorer=lambda samples, _sr: float(np.mean(samples)),
    )
    packets = []
    for sequence, value, timestamp in (
        (40, 0.1, 8.25),
        (41, 0.2, 8.55),
        (42, 0.3, 8.85),
    ):
        packets.append(
            source.update(
                audio_window=_audio_window(value, t_start=timestamp - 0.25),
                force_sample=_force_sample(t=timestamp - 0.05),
                inference_sequence=sequence,
                inference_monotonic_s=timestamp,
                observed_monotonic_s=timestamp + 0.01,
            )["packet"]
        )

    writer = realtime_q.QualityPacketLogWriter(target_dir)
    first_claim = packets[0]["audio_window"]["model_input"]
    first_raw = (
        source.packet_log_writer.input_staging_path
        / Path(str(first_claim["path"])).name
    ).read_bytes()
    writer.persist_model_input(inference_sequence=40, raw=first_raw)
    writer.append(packets[0])
    with pytest.raises(realtime_q.QualityPacketError, match="chain|sequence"):
        last_claim = packets[2]["audio_window"]["model_input"]
        last_raw = (
            source.packet_log_writer.input_staging_path
            / Path(str(last_claim["path"])).name
        ).read_bytes()
        writer.persist_model_input(inference_sequence=42, raw=last_raw)


def test_quality_packet_writer_refuses_degenerate_output_and_keeps_staging(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = _production_quality(
        tmp_path,
        scorer=lambda _samples, _sr: 0.5,
        packet_log_writer=writer,
    )
    packets = []
    for sequence, value, timestamp in (
        (50, 0.1, 9.25),
        (51, 0.2, 9.55),
        (52, 0.3, 9.85),
    ):
        packets.append(quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )["packet"])
    receipt = quality.quality_authenticity(now_monotonic_s=9.9)
    _write_force_csv(tmp_path, packets)

    with pytest.raises(realtime_q.QualityPacketError, match="nondegenerate"):
        writer.finalize(receipt, now_monotonic_s=9.9)

    assert (tmp_path / ".quality_inference_packets.jsonl.inprogress").is_file()
    assert not (tmp_path / "quality_inference_packets.jsonl").exists()


def test_quality_packet_writer_rejects_a_truncated_persisted_line(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = _production_quality(
        tmp_path,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    for sequence, value, timestamp in (
        (60, 0.1, 10.25),
        (61, 0.2, 10.55),
    ):
        quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )
    receipt = quality.quality_authenticity(now_monotonic_s=10.6)
    with writer.staging_path.open("ab") as stream:
        stream.write(b'{"truncated":')

    with pytest.raises(realtime_q.QualityPacketError, match="incomplete|invalid JSON"):
        writer.finalize(receipt, now_monotonic_s=10.6)

    assert not writer.final_path.exists()


def test_production_scorer_consumes_truly_immutable_float32_le_bytes(tmp_path):
    observed = []

    def scorer(samples, _sr):
        observed.append(samples.tobytes(order="C"))
        assert samples.dtype.str == "<f4"
        assert samples.flags.writeable is False
        with pytest.raises(ValueError):
            samples.setflags(write=True)
        return float(np.mean(samples))

    quality = _production_quality(tmp_path, scorer=scorer)
    window = _audio_window(0.125, t_start=30.0)
    result = quality.update(
        audio_window=window,
        force_sample=_force_sample(t=30.2),
        inference_sequence=1,
        inference_monotonic_s=30.25,
        observed_monotonic_s=30.26,
    )

    expected = np.asarray(window.samples, dtype="<f4").tobytes(order="C")
    assert observed == [expected]
    assert result["packet"]["audio_window"]["model_input"]["sha256"] == (
        hashlib.sha256(expected).hexdigest()
    )


def test_diagnostic_proxy_cannot_own_a_production_input_ledger(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    with pytest.raises(ValueError, match="diagnostic.*production.*ledger"):
        realtime_q.StreamingQuality(
            scorer=realtime_q.band_energy_proxy,
            mode="diagnostic",
            policy=realtime_q.diagnostic_quality_policy(
                window_s=0.25,
                force_band=(3.0, 5.0),
            ),
            packet_log_writer=writer,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("raw", "raw ADC"),
        ("hall_v", "Hall voltage"),
        ("displacement_mm", "displacement"),
        ("device_t_ms", "device timestamp"),
    ],
)
def test_missing_exact_force_csv_field_is_rejected_before_scorer(
    tmp_path, field, message,
):
    calls = []
    quality = _production_quality(
        tmp_path,
        scorer=lambda _samples, _sr: calls.append("called") or 0.5,
    )
    sample = _force_sample(t=31.2)
    delattr(sample, field)

    with pytest.raises(realtime_q.QualityPacketError, match=message):
        quality.update(
            audio_window=_audio_window(0.1, t_start=31.0),
            force_sample=sample,
            inference_sequence=1,
            inference_monotonic_s=31.25,
            observed_monotonic_s=31.26,
        )

    assert calls == []


def test_model_input_writer_rejects_duplicate_replay_and_gap(tmp_path):
    first_dir = tmp_path / "duplicate"
    first_dir.mkdir()
    writer = realtime_q.QualityPacketLogWriter(first_dir)
    raw = np.arange(8, dtype="<f4").tobytes()
    writer.persist_model_input(inference_sequence=7, raw=raw)
    with pytest.raises(realtime_q.QualityPacketError, match="duplicate|replay"):
        writer.persist_model_input(inference_sequence=7, raw=raw)

    second_dir = tmp_path / "gap"
    second_dir.mkdir()
    gap_writer = realtime_q.QualityPacketLogWriter(second_dir)
    first = _production_quality(
        second_dir,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=gap_writer,
    )
    first.update(
        audio_window=_audio_window(0.1, t_start=32.0),
        force_sample=_force_sample(t=32.2),
        inference_sequence=7,
        inference_monotonic_s=32.25,
        observed_monotonic_s=32.26,
    )
    with pytest.raises(realtime_q.QualityPacketError, match="sequence"):
        first.update(
            audio_window=_audio_window(0.2, t_start=32.3),
            force_sample=_force_sample(t=32.5),
            inference_sequence=9,
            inference_monotonic_s=32.55,
            observed_monotonic_s=32.56,
        )


@pytest.mark.parametrize(
    "fault",
    ["missing", "extra", "symlink", "content", "swapped"],
)
def test_persisted_verifier_rejects_model_input_inventory_faults(
    tmp_path, fault,
):
    _finalized_quality_session(tmp_path)
    root = tmp_path / "quality_model_inputs"
    files = sorted(root.iterdir())
    if fault == "missing":
        files[0].unlink()
    elif fault == "extra":
        (root / "unclaimed.f32le").write_bytes(b"\x00\x00\x00\x00")
    elif fault == "symlink":
        files[0].unlink()
        files[0].symlink_to(files[1])
    elif fault == "content":
        raw = bytearray(files[0].read_bytes())
        raw[0] ^= 0x01
        files[0].write_bytes(raw)
    else:
        first = files[0].read_bytes()
        second = files[1].read_bytes()
        files[0].write_bytes(second)
        files[1].write_bytes(first)

    with pytest.raises(
        realtime_q.QualityPacketError,
        match="model input|inventory|content|regular non-symlink",
    ):
        realtime_q.verify_quality_packet_log(tmp_path)


def test_persisted_verifier_rejects_residual_model_input_staging(tmp_path):
    _finalized_quality_session(tmp_path)
    (tmp_path / ".quality_model_inputs.inprogress").mkdir()

    with pytest.raises(realtime_q.QualityPacketError, match="staging"):
        realtime_q.verify_quality_packet_log(tmp_path)


def test_receipt_inventory_tamper_is_rejected(tmp_path):
    _finalized_quality_session(tmp_path)
    receipt_path = tmp_path / "quality_inference_authenticity.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["model_input_inventory"]["total_bytes"] += 4
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(realtime_q.QualityPacketError, match="full-chain"):
        realtime_q.verify_quality_packet_log(tmp_path)


def test_force_csv_binding_rejects_self_consistent_packet_value_not_in_csv(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    writer = realtime_q.QualityPacketLogWriter(session)
    quality = _production_quality(
        session,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    packets = [quality.update(
        audio_window=_audio_window(0.1, t_start=40.0),
        force_sample=_force_sample(t=40.2),
        inference_sequence=1,
        inference_monotonic_s=40.25,
        observed_monotonic_s=40.26,
    )["packet"]]
    force_csv = _write_force_csv(session, packets)
    tampered = realtime_q._json_clone(packets)
    tampered[0]["force_sample"]["force_n"] = 4.25
    _rehash_packet_chain(tampered)

    with pytest.raises(realtime_q.QualityPacketError, match="force.csv.*match"):
        realtime_q.verify_quality_force_csv_binding(tampered, force_csv)


def test_force_csv_binding_rejects_duplicate_timestamp_ambiguity(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = _production_quality(
        tmp_path,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    packets = [quality.update(
        audio_window=_audio_window(0.1, t_start=41.0),
        force_sample=_force_sample(t=41.2),
        inference_sequence=1,
        inference_monotonic_s=41.25,
        observed_monotonic_s=41.26,
    )["packet"]]
    force_csv = _write_force_csv(tmp_path, packets)
    with force_csv.open("a", encoding="utf-8") as stream:
        stream.write(_force_csv_row(packets[0]["force_sample"]) + "\n")

    with pytest.raises(
        realtime_q.QualityPacketError,
        match="duplicate|ambiguous|repeated",
    ):
        realtime_q.verify_quality_force_csv_binding(packets, force_csv)


def test_persisted_verifier_rejects_force_csv_tamper(tmp_path):
    _finalized_quality_session(tmp_path)
    force_csv = tmp_path / "force.csv"
    lines = force_csv.read_text(encoding="utf-8").splitlines()
    first = lines[1].split(",")
    first[4] = format(float(first[4]) + 0.25, ".17g")
    lines[1] = ",".join(first)
    force_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(realtime_q.QualityPacketError, match="force.csv|force input"):
        realtime_q.verify_quality_packet_log(tmp_path)


@pytest.mark.parametrize("fault", ["dtype", "path", "shape", "bytes", "sha256"])
def test_full_chain_rejects_tampered_model_input_claims(tmp_path, fault):
    _, log_receipt = _finalized_quality_session(tmp_path)
    packets = realtime_q.read_quality_packet_log(
        tmp_path / "quality_inference_packets.jsonl"
    )
    tampered = realtime_q._json_clone(packets)
    claim = tampered[0]["audio_window"]["model_input"]
    if fault == "dtype":
        claim["dtype"] = ">f4"
    elif fault == "path":
        claim["path"] = "quality_model_inputs/input_00000000000000000101.f32le"
    elif fault == "shape":
        claim["shape"] = [3_999]
        claim["bytes"] = 3_999 * 4
    elif fault == "bytes":
        claim["bytes"] -= 4
    else:
        claim["sha256"] = "f" * 64
    _rehash_packet_chain(tampered)
    authenticity = realtime_q._json_clone(log_receipt["quality_authenticity"])
    authenticity["last_packet"] = realtime_q._json_clone(tampered[-1])
    authenticity["last_packet_sha256"] = tampered[-1]["packet_sha256"]
    authenticity["packet_chain_sha256"] = tampered[-1]["packet_chain_sha256"]

    with pytest.raises(realtime_q.QualityPacketError, match="model input"):
        realtime_q.verify_quality_packet_chain(
            tampered,
            authenticity,
            now_monotonic_s=20.9,
            model_input_dir=tmp_path / "quality_model_inputs",
            force_csv_path=tmp_path / "force.csv",
        )


def test_force_csv_binding_rejects_one_row_reused_by_two_packets(tmp_path):
    _finalized_quality_session(tmp_path)
    packets = realtime_q.read_quality_packet_log(
        tmp_path / "quality_inference_packets.jsonl"
    )[:2]
    tampered = realtime_q._json_clone(packets)
    tampered[1]["force_sample"] = realtime_q._json_clone(
        tampered[0]["force_sample"]
    )
    _rehash_packet_chain(tampered)

    with pytest.raises(realtime_q.QualityPacketError, match="duplicate|replay"):
        realtime_q.verify_quality_force_csv_binding(
            tampered,
            tmp_path / "force.csv",
        )


def test_quality_packet_writer_refuses_publication_without_force_csv(tmp_path):
    writer = realtime_q.QualityPacketLogWriter(tmp_path)
    quality = _production_quality(
        tmp_path,
        scorer=lambda samples, _sr: float(np.mean(samples)),
        packet_log_writer=writer,
    )
    for sequence, value, timestamp in (
        (200, 0.1, 50.25),
        (201, 0.2, 50.55),
        (202, 0.3, 50.85),
    ):
        quality.update(
            audio_window=_audio_window(value, t_start=timestamp - 0.25),
            force_sample=_force_sample(t=timestamp - 0.05),
            inference_sequence=sequence,
            inference_monotonic_s=timestamp,
            observed_monotonic_s=timestamp + 0.01,
        )
    receipt = quality.quality_authenticity(now_monotonic_s=50.9)

    with pytest.raises(realtime_q.QualityPacketError, match="force.csv"):
        writer.finalize(receipt, now_monotonic_s=50.9)

    assert writer.staging_path.is_file()
    assert writer.input_staging_path.is_dir()
    assert not writer.final_path.exists()


# --- Q raw-source provenance validator (Step 2b) -----------------------------

def _raw_source(**overrides):
    base = {
        "source_channel": 1,
        "source_sample_start": 0,
        "source_sample_end": 12_000,
        "source_sample_rate": 48_000,
        "resampler_identity": {
            "algorithm": "polyphase_fir_v1",
            "input_sample_rate": 48_000,
            "output_sample_rate": 16_000,
        },
    }
    base.update(overrides)
    return base


def test_raw_source_claim_round_trips_and_binds_window_rate():
    out = realtime_q._normalize_raw_source_claim(_raw_source(), window_sample_rate=16_000)
    assert out["schema_version"] == "wam_quality_audio_raw_source_v1"
    # Re-validating the stored form (with schema_version) is idempotent.
    assert realtime_q._normalize_raw_source_claim(out, window_sample_rate=16_000) == out


@pytest.mark.parametrize("claim, match", [
    (_raw_source(source_channel=-1), "channel"),
    (_raw_source(source_sample_start=5, source_sample_end=5), "sample range"),
    (_raw_source(source_sample_start=10, source_sample_end=3), "sample range"),
    (_raw_source(source_sample_rate=0), "sample rate"),
    (_raw_source(resampler_identity={"algorithm": "x", "input_sample_rate": 48_000}),
     "exact schema"),
    (_raw_source(resampler_identity={
        "algorithm": "", "input_sample_rate": 48_000, "output_sample_rate": 16_000}),
     "algorithm"),
    (_raw_source(resampler_identity={
        "algorithm": "x", "input_sample_rate": 44_100, "output_sample_rate": 16_000}),
     "input rate must equal the source"),
])
def test_raw_source_claim_rejects_bad_provenance(claim, match):
    with pytest.raises(ValueError, match=match):
        realtime_q._normalize_raw_source_claim(claim, window_sample_rate=16_000)


def test_raw_source_claim_rejects_window_rate_mismatch():
    with pytest.raises(ValueError, match="output rate must equal the window"):
        realtime_q._normalize_raw_source_claim(_raw_source(), window_sample_rate=8_000)


def test_raw_source_claim_rejects_missing_or_extra_fields():
    with pytest.raises(ValueError, match="exact schema"):
        realtime_q._normalize_raw_source_claim({"source_channel": 1})
    with pytest.raises(ValueError, match="exact schema"):
        realtime_q._normalize_raw_source_claim(_raw_source(unexpected=1))
