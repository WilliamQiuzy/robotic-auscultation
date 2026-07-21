from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import pytest

from ausc.perception.robot.camera_exposure_evidence import (
    CameraExposureContract,
    CameraExposureEvidence,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _artifact(**changes) -> dict:
    value = {
        "schema_version": "wam_camera_latency_v1",
        "latency_definition": "host_arrival_minus_exposure",
        "latency_ms": 18.0,
        "latency_std_ms": 2.0,
        "n_samples": 40,
        "measurement_method": "hardware_led_flash",
        "capture_mode": {
            "backend": "orbbec",
            "camera_id": "CL8326300A0",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "time_source": "orbbec_device_and_host_arrival",
            "color_format": "MJPG",
            "depth_mode": {
                "width": 640,
                "height": 576,
                "fps": 30.0,
                "format": "Y16",
            },
            "align_mode": "software_depth_to_color",
            "frame_aggregate_mode": "full_frame_require",
        },
    }
    value.update(changes)
    return value


def _digest(value: dict) -> str:
    raw = (json.dumps(value, indent=2) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _contract(**artifact_changes) -> CameraExposureContract:
    artifact = _artifact(**artifact_changes)
    return CameraExposureContract.from_artifact(
        artifact,
        artifact_sha256=_digest(artifact),
        camera_profile_sha256=SHA_A,
        alignment_sha256=SHA_B,
        expected_camera_id="CL8326300A0",
        expected_capture_mode=artifact["capture_mode"],
    )


def test_frame_exposure_interval_is_derived_on_clock_monotonic_before_fanout() -> None:
    evidence = _contract().for_frame(
        frame_index=17,
        camera_host_t_s=10.0,
        camera_device_t_ms=1234.5,
    )

    assert isinstance(evidence, CameraExposureEvidence)
    assert evidence.frame_index == 17
    assert evidence.camera_host_monotonic_ns == 10_000_000_000
    assert evidence.exposure_nominal_monotonic_ns == 9_982_000_000
    assert evidence.exposure_lower_monotonic_ns == 9_976_000_000
    assert evidence.exposure_upper_monotonic_ns == 9_988_000_000
    assert (
        evidence.exposure_upper_monotonic_ns
        - evidence.exposure_lower_monotonic_ns
        == 12_000_000
    )
    assert evidence.camera_profile_sha256 == SHA_A
    assert evidence.alignment_sha256 == SHA_B


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"measurement_method": "software_estimate"}, "hardware LED"),
        ({"latency_definition": "unknown"}, "latency definition"),
        ({"latency_ms": 101.0}, "100 ms"),
        ({"latency_std_ms": 0.0}, "standard deviation"),
        ({"latency_std_ms": 7.0}, "40 ms"),
        ({"n_samples": 19}, "samples"),
    ),
)
def test_uncommissioned_or_too_uncertain_timing_artifact_is_rejected(
    changes: dict, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _contract(**changes)


def test_artifact_hash_mode_camera_and_alignment_are_exactly_bound() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="SHA-256"):
        CameraExposureContract.from_artifact(
            artifact,
            artifact_sha256="0" * 64,
            camera_profile_sha256=SHA_A,
            alignment_sha256=SHA_B,
            expected_camera_id="CL8326300A0",
            expected_capture_mode=artifact["capture_mode"],
        )
    wrong_mode = dict(artifact["capture_mode"])
    wrong_mode["fps"] = 15.0
    with pytest.raises(ValueError, match="capture mode"):
        CameraExposureContract.from_artifact(
            artifact,
            artifact_sha256=_digest(artifact),
            camera_profile_sha256=SHA_A,
            alignment_sha256=SHA_B,
            expected_camera_id="CL8326300A0",
            expected_capture_mode=wrong_mode,
        )
    with pytest.raises(ValueError, match="camera"):
        CameraExposureContract.from_artifact(
            artifact,
            artifact_sha256=_digest(artifact),
            camera_profile_sha256=SHA_A,
            alignment_sha256=SHA_B,
            expected_camera_id="wrong-camera",
            expected_capture_mode=artifact["capture_mode"],
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"frame_index": -1},
        {"camera_host_t_s": float("nan")},
        {"camera_device_t_ms": None},
    ),
)
def test_missing_or_invalid_frame_clock_fails_closed(changes: dict) -> None:
    values = {
        "frame_index": 17,
        "camera_host_t_s": 10.0,
        "camera_device_t_ms": 1234.5,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        _contract().for_frame(**values)


def test_frame_evidence_is_immutable() -> None:
    evidence = _contract().for_frame(
        frame_index=17,
        camera_host_t_s=10.0,
        camera_device_t_ms=1234.5,
    )
    with pytest.raises(FrozenInstanceError):
        evidence.frame_index = 18
