"""Quality-gated persistence tests for measured camera transport latency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

AUSC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUSC))

import camera_latency_calib as cal  # noqa: E402


MODE = {
    "backend": "orbbec",
    "camera_id": "femto-bolt-01",
    "width": 640,
    "height": 480,
    "fps": 30.0,
    "time_source": "orbbec_device_and_host_arrival",
    "color_format": "RGB",
    "depth_mode": {"width": 640, "height": 576, "fps": 30.0, "format": "Y16"},
    "align_mode": "software_depth_to_color",
    "frame_aggregate_mode": "full_frame_require",
}


def _persist(path, samples):
    return cal.persist_camera_latency(path, samples, **MODE)


def test_persist_camera_latency_requires_enough_low_jitter_physical_samples(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        _persist(tmp_path / "camera.json", [12.0] * 19)
    with pytest.raises(ValueError, match="standard deviation"):
        _persist(
            tmp_path / "camera.json", np.linspace(1.0, 30.0, 20).tolist(),
        )
    assert not (tmp_path / "camera.json").exists()


def test_persist_camera_latency_writes_versioned_quality_contract(tmp_path):
    path = tmp_path / "camera.json"
    payload = _persist(path, [11.0 + (i % 3) for i in range(20)])
    saved = json.loads(path.read_text())

    assert saved == payload
    assert saved["schema_version"] == "wam_camera_latency_v1"
    assert saved["latency_definition"] == "host_arrival_minus_exposure"
    assert saved["measurement_method"] == "hardware_led_flash"
    assert saved["n_samples"] == 20
    assert saved["latency_ms"] == pytest.approx(np.mean([11.0 + (i % 3) for i in range(20)]))
    assert saved["latency_std_ms"] <= 5.0
    assert saved["capture_mode"] == MODE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", ""),
        ("camera_id", ""),
        ("width", 0),
        ("height", 0),
        ("fps", float("nan")),
        ("time_source", "opencv_host_arrival"),
        ("color_format", ""),
        ("depth_mode", None),
        ("align_mode", "unknown"),
        ("frame_aggregate_mode", "disabled"),
    ],
)
def test_camera_latency_requires_concrete_capture_mode_identity(tmp_path, field, value):
    mode = {**MODE, field: value}
    with pytest.raises(ValueError, match=f"capture_mode.*{field}"):
        cal.persist_camera_latency(tmp_path / "camera.json", [12.0] * 20, **mode)


def test_camera_latency_cli_requires_explicit_mode_identity(tmp_path):
    with pytest.raises(SystemExit):
        cal.main([
            "--samples-ms", *(["12"] * 20), "--out", str(tmp_path / "camera.json"),
        ])
