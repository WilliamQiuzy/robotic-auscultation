"""Mode-bound camera exposure intervals on the host monotonic clock."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NS_PER_MS = 1_000_000
_NS_PER_SECOND = 1_000_000_000
_UNCERTAINTY_SIGMA = 3.0
_MAX_CAMERA_TO_EXPOSURE_NS = 100_000_000
_MAX_EXPOSURE_INTERVAL_NS = 40_000_000
_MIN_SAMPLES = 20
_ACCEPTED_TIMING_METHODS = frozenset({"hardware_led_flash", "device_timer_host_sync"})


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _artifact_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(value), indent=2) + "\n").encode("utf-8")


def _semantic_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CameraExposureEvidence:
    frame_index: int
    camera_id: str
    camera_host_t_s: float
    camera_host_monotonic_ns: int
    camera_device_t_ms: float
    exposure_lower_host_t_s: float
    exposure_nominal_host_t_s: float
    exposure_upper_host_t_s: float
    exposure_lower_monotonic_ns: int
    exposure_nominal_monotonic_ns: int
    exposure_upper_monotonic_ns: int
    camera_latency_sha256: str
    camera_profile_sha256: str
    alignment_sha256: str


@dataclass(frozen=True)
class CameraExposureContract:
    camera_id: str
    camera_latency_sha256: str
    camera_profile_sha256: str
    alignment_sha256: str
    latency_ns: int
    latency_uncertainty_ns: int
    capture_mode_json: str
    measurement_method: str
    n_samples: int

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, object],
        *,
        artifact_sha256: str,
        camera_profile_sha256: str,
        alignment_sha256: str,
        expected_camera_id: str,
        expected_capture_mode: Mapping[str, object],
    ) -> "CameraExposureContract":
        if not isinstance(artifact, Mapping):
            raise ValueError("camera latency artifact must be a JSON object")
        actual_digest = hashlib.sha256(_artifact_bytes(artifact)).hexdigest()
        if (
            not isinstance(artifact_sha256, str)
            or _SHA256.fullmatch(artifact_sha256) is None
            or artifact_sha256 != actual_digest
        ):
            raise ValueError("camera latency artifact SHA-256 does not match bytes")
        for digest, label in (
            (camera_profile_sha256, "camera profile"),
            (alignment_sha256, "alignment"),
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{label} SHA-256 is invalid")
        if artifact.get("schema_version") != "wam_camera_latency_v1":
            raise ValueError("camera latency artifact schema is unsupported")
        if artifact.get("latency_definition") != "host_arrival_minus_exposure":
            raise ValueError("camera latency definition is unsupported")
        # Accepted hardware timing methods. "hardware_led_flash" pins exposure to a known host
        # time via an in-view LED. "device_timer_host_sync" instead uses the camera's own hardware
        # timestamp synchronised to the host clock (Orbbec timer_sync_with_host): per-frame
        # latency = host system_timestamp - device exposure_timestamp, both in the host domain.
        # It is a hardware-clock measurement (no invented/software time) and, on the Femto Bolt,
        # is highly repeatable (sub-ms std), so it is admitted alongside the LED method.
        if artifact.get("measurement_method") not in _ACCEPTED_TIMING_METHODS:
            raise ValueError("camera timing requires a hardware timing measurement")
        capture_mode = artifact.get("capture_mode")
        if not isinstance(capture_mode, Mapping):
            raise ValueError("camera latency capture mode is missing")
        if _semantic_json(capture_mode) != _semantic_json(expected_capture_mode):
            raise ValueError("camera latency capture mode does not match active camera")
        camera_id = capture_mode.get("camera_id")
        if (
            not isinstance(expected_camera_id, str)
            or not expected_camera_id
            or camera_id != expected_camera_id
        ):
            raise ValueError("camera identity does not match timing artifact")

        latency_ms = artifact.get("latency_ms")
        std_ms = artifact.get("latency_std_ms")
        n_samples = artifact.get("n_samples")
        if not _finite_number(latency_ms) or not 0.0 < float(latency_ms) <= 100.0:
            raise ValueError("camera latency must be within the 100 ms safety envelope")
        if not _finite_number(std_ms) or float(std_ms) <= 0.0:
            raise ValueError("camera latency standard deviation must be positive")
        if type(n_samples) is not int or n_samples < _MIN_SAMPLES:
            raise ValueError("camera latency artifact has too few hardware samples")
        uncertainty_ms = _UNCERTAINTY_SIGMA * float(std_ms)
        if 2.0 * uncertainty_ms > 40.0:
            raise ValueError("camera exposure interval cannot exceed 40 ms")
        if float(latency_ms) <= uncertainty_ms:
            raise ValueError("camera latency uncertainty reaches host arrival")

        latency_ns = int(round(float(latency_ms) * _NS_PER_MS))
        uncertainty_ns = int(round(uncertainty_ms * _NS_PER_MS))
        if (
            latency_ns + uncertainty_ns > _MAX_CAMERA_TO_EXPOSURE_NS
            or 2 * uncertainty_ns > _MAX_EXPOSURE_INTERVAL_NS
        ):
            raise ValueError("camera timing exceeds the production safety envelope")
        return cls(
            camera_id=str(camera_id),
            camera_latency_sha256=artifact_sha256,
            camera_profile_sha256=camera_profile_sha256,
            alignment_sha256=alignment_sha256,
            latency_ns=latency_ns,
            latency_uncertainty_ns=uncertainty_ns,
            capture_mode_json=_semantic_json(capture_mode),
            measurement_method=str(artifact.get("measurement_method")),
            n_samples=n_samples,
        )

    def for_frame(
        self,
        *,
        frame_index: int,
        camera_host_t_s: float,
        camera_device_t_ms: float,
    ) -> CameraExposureEvidence:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("camera frame index must be a nonnegative integer")
        if not _finite_number(camera_host_t_s) or float(camera_host_t_s) <= 0.0:
            raise ValueError("camera host monotonic time is invalid")
        if not _finite_number(camera_device_t_ms):
            raise ValueError("camera device timestamp is required")
        host_ns = int(round(float(camera_host_t_s) * _NS_PER_SECOND))
        nominal_ns = host_ns - self.latency_ns
        lower_ns = nominal_ns - self.latency_uncertainty_ns
        upper_ns = nominal_ns + self.latency_uncertainty_ns
        if (
            not 0 < lower_ns < nominal_ns < upper_ns < host_ns
            or upper_ns - lower_ns > _MAX_EXPOSURE_INTERVAL_NS
            or host_ns - lower_ns > _MAX_CAMERA_TO_EXPOSURE_NS
        ):
            raise ValueError("derived camera exposure interval is unsafe")
        return CameraExposureEvidence(
            frame_index=frame_index,
            camera_id=self.camera_id,
            camera_host_t_s=float(camera_host_t_s),
            camera_host_monotonic_ns=host_ns,
            camera_device_t_ms=float(camera_device_t_ms),
            exposure_lower_host_t_s=lower_ns / _NS_PER_SECOND,
            exposure_nominal_host_t_s=nominal_ns / _NS_PER_SECOND,
            exposure_upper_host_t_s=upper_ns / _NS_PER_SECOND,
            exposure_lower_monotonic_ns=lower_ns,
            exposure_nominal_monotonic_ns=nominal_ns,
            exposure_upper_monotonic_ns=upper_ns,
            camera_latency_sha256=self.camera_latency_sha256,
            camera_profile_sha256=self.camera_profile_sha256,
            alignment_sha256=self.alignment_sha256,
        )


__all__ = [
    "CameraExposureContract",
    "CameraExposureEvidence",
]
