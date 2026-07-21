"""Hardware-free contract tests for the Lingbox Orbbec hand-eye preflight."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import handeye_orbbec_preflight as preflight


def _intrinsics(width: int = 20, height: int = 20) -> dict[str, float | int]:
    return {
        "fx": 200.0,
        "fy": 210.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "width": width,
        "height": height,
    }


def _sdk_xyz_for_depth(
    depth_raw: np.ndarray,
    intrinsics: dict[str, float | int],
    *,
    depth_scale_mm: float = 1.0,
) -> np.ndarray:
    height, width = depth_raw.shape
    y, x = np.indices((height, width), dtype=np.float64)
    z = depth_raw.astype(np.float64) * depth_scale_mm
    return np.stack(
        [
            (x - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
            (y - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
            z,
        ],
        axis=-1,
    )


def _measurement(
    *,
    width: int = 20,
    height: int = 20,
    sdk_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    depth_mask: np.ndarray | None = None,
    nonfinite: bool = False,
):
    intrinsics = _intrinsics(width, height)
    depth = np.full((height, width), 1000, dtype=np.uint16)
    if depth_mask is not None:
        depth = np.where(depth_mask, depth, 0).astype(np.uint16)
    sdk_xyz_mm = _sdk_xyz_for_depth(depth, intrinsics)
    sdk_xyz_mm += np.asarray(sdk_offset_mm, dtype=np.float64)
    if nonfinite:
        sdk_xyz_mm[0, 0, 0] = np.nan
    # Exercise the SDK coordinate-value scale instead of assuming raw floats
    # are already millimetres.
    return preflight.compare_aligned_depth_to_sdk_xyz(
        depth,
        depth_scale_mm=1.0,
        sdk_xyz_raw=sdk_xyz_mm / 0.5,
        sdk_position_scale_mm=0.5,
        intrinsics=intrinsics,
        sample_stride=1,
    )


def test_geometry_gate_accepts_three_finite_wide_coverage_frames():
    result = preflight.evaluate_geometry_gate([_measurement() for _ in range(3)])

    assert result["status"] == "PASS"
    assert result["frames"]["count"] == 3
    assert result["samples"]["valid_min_per_frame"] == 400
    assert result["samples"]["valid_total"] == 1200
    assert result["coverage"]["x_fraction_min"] == pytest.approx(1.0)
    assert result["coverage"]["y_fraction_min"] == pytest.approx(1.0)
    assert result["errors_mm"]["p95"] == pytest.approx(0.0, abs=1e-9)
    assert result["errors_mm"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert result["failures"] == []


@pytest.mark.parametrize(
    ("measurements", "failure"),
    [
        (lambda: [_measurement() for _ in range(2)], "frames_below_3"),
        (
            lambda: [_measurement(width=9, height=11) for _ in range(3)],
            "valid_samples_below_100",
        ),
        (
            lambda: [
                _measurement(
                    depth_mask=np.indices((20, 20))[1] < 10,
                )
                for _ in range(3)
            ],
            "x_coverage_below_0.6",
        ),
        (
            lambda: [_measurement(nonfinite=True) for _ in range(3)],
            "nonfinite_correspondences",
        ),
        (
            lambda: [
                _measurement(sdk_offset_mm=(2.1, 0.0, 0.0)) for _ in range(3)
            ],
            "p95_error_above_2mm",
        ),
    ],
)
def test_geometry_gate_fails_closed_for_each_required_threshold(measurements, failure):
    result = preflight.evaluate_geometry_gate(measurements())

    assert result["status"] == "FAIL"
    assert failure in result["failures"]


def test_geometry_gate_rejects_single_maximum_error_above_5mm():
    measurement = _measurement()
    measurement.errors_mm[0] = 5.01

    result = preflight.evaluate_geometry_gate([measurement for _ in range(3)])

    assert result["errors_mm"]["p95"] < 2.0
    assert result["status"] == "FAIL"
    assert "max_error_above_5mm" in result["failures"]


def test_comparison_rejects_shape_or_intrinsics_mismatch():
    depth = np.ones((20, 20), dtype=np.uint16)
    xyz = np.ones((20, 20, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="SDK XYZ shape"):
        preflight.compare_aligned_depth_to_sdk_xyz(
            depth,
            depth_scale_mm=1.0,
            sdk_xyz_raw=xyz[:19],
            sdk_position_scale_mm=1.0,
            intrinsics=_intrinsics(),
        )
    with pytest.raises(ValueError, match="intrinsics resolution"):
        preflight.compare_aligned_depth_to_sdk_xyz(
            depth,
            depth_scale_mm=1.0,
            sdk_xyz_raw=xyz,
            sdk_position_scale_mm=1.0,
            intrinsics=_intrinsics(19, 20),
        )


def test_profile_calibration_records_exact_k_and_all_eight_native_coefficients():
    distortion = SimpleNamespace(
        k1=0.01,
        k2=0.02,
        k3=0.03,
        k4=0.04,
        k5=0.05,
        k6=0.06,
        p1=0.07,
        p2=0.08,
    )
    profile = SimpleNamespace(
        get_width=lambda: 1920,
        get_height=lambda: 1080,
        get_intrinsic=lambda: SimpleNamespace(
            fx=1001.0, fy=1002.0, cx=959.5, cy=539.5,
        ),
        get_distortion=lambda: distortion,
    )

    result = preflight.extract_active_profile_calibration(profile)

    assert result["K"] == [
        [1001.0, 0.0, 959.5],
        [0.0, 1002.0, 539.5],
        [0.0, 0.0, 1.0],
    ]
    assert result["intrinsics"] == {
        "fx": 1001.0,
        "fy": 1002.0,
        "cx": 959.5,
        "cy": 539.5,
        "width": 1920,
        "height": 1080,
    }
    assert result["distortion"] == {
        "model": "model_api_unexposed",
        "coefficient_order": ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
        "coefficients": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    }


def test_sample_one_aligned_frame_uses_same_frameset_for_depth_and_sdk_points():
    intrinsics = _intrinsics()
    depth_raw = np.full((20, 20), 1000, dtype=np.uint16)
    xyz_mm = _sdk_xyz_for_depth(depth_raw, intrinsics).astype(np.float32)
    depth_frame = SimpleNamespace(
        get_width=lambda: 20,
        get_height=lambda: 20,
        get_depth_scale=lambda: 1.0,
        get_data=lambda: depth_raw.tobytes(),
    )
    aligned = SimpleNamespace(
        get_depth_frame=lambda: depth_frame,
        get_color_frame=lambda: object(),
    )
    raw_frameset = object()
    pipeline = SimpleNamespace(wait_for_frames=lambda _timeout: raw_frameset)

    class Align:
        def process(self, value):
            assert value is raw_frameset
            return aligned

    points = SimpleNamespace(
        get_width=lambda: 20,
        get_height=lambda: 20,
        get_position_value_scale=lambda: 0.5,
        get_data=lambda: (xyz_mm / 0.5).astype(np.float32).tobytes(),
    )
    point_frame = SimpleNamespace(as_points_frame=lambda: points)

    class PointCloud:
        def process(self, value):
            assert value is aligned
            return point_frame

    result = preflight.sample_one_aligned_frame(
        pipeline,
        Align(),
        PointCloud(),
        intrinsics,
        sample_stride=1,
    )

    assert result.valid_sample_count == 400
    assert np.max(result.errors_mm) < 1e-3


class _Profile:
    def __init__(self, width, height, fps, pixel_format):
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_fps(self):
        return self.fps

    def get_format(self):
        return self.pixel_format

    def as_video_stream_profile(self):
        return self

    def get_intrinsic(self):
        return SimpleNamespace(fx=1001.0, fy=1002.0, cx=959.5, cy=539.5)

    def get_distortion(self):
        return SimpleNamespace(
            k1=0.01, k2=0.02, k3=0.03, k4=0.04,
            k5=0.05, k6=0.06, p1=0.07, p2=0.08,
        )


def _fake_sdk(calls: dict[str, object]):
    color_profile = _Profile(1920, 1080, 30, "mjpg")
    depth_profile = _Profile(640, 576, 30, "y16")

    class Profiles:
        def __init__(self, sensor):
            self.sensor = sensor

        def get_video_stream_profile(self, width, height, pixel_format, fps):
            calls.setdefault("profile_requests", []).append(
                (self.sensor, width, height, pixel_format, fps)
            )
            return color_profile if self.sensor == "color" else depth_profile

    info = SimpleNamespace(
        get_serial_number=lambda: "CL8326300A0",
        get_name=lambda: "Femto Bolt",
        get_firmware_version=lambda: "1.2.3",
    )
    device = SimpleNamespace(get_device_info=lambda: info)

    class Devices:
        def get_device_by_serial_number(self, serial):
            calls["serial_lookup"] = serial
            return device

    class Context:
        def query_devices(self):
            return Devices()

    class Pipeline:
        def __init__(self, selected_device):
            calls["pipeline_device"] = selected_device

        def get_stream_profile_list(self, sensor):
            return Profiles(sensor)

        def start(self, config):
            calls["started"] = config

        def wait_for_frames(self, _timeout):
            raise AssertionError("the injected sampler should be used")

        def stop(self):
            calls["stop_count"] = int(calls.get("stop_count", 0)) + 1

    class Config:
        def __init__(self):
            self.enabled = []
            self.aggregate_mode = None

        def enable_stream(self, profile):
            self.enabled.append(profile)

        def set_frame_aggregate_output_mode(self, mode):
            self.aggregate_mode = mode

    class AlignFilter:
        def __init__(self, align_to_stream):
            calls["align_target"] = align_to_stream

    class PointCloudFilter:
        def set_create_point_format(self, value):
            calls["point_format"] = value

    return SimpleNamespace(
        Context=Context,
        Pipeline=Pipeline,
        Config=Config,
        AlignFilter=AlignFilter,
        PointCloudFilter=PointCloudFilter,
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth"),
        OBFormat=SimpleNamespace(MJPG="mjpg", Y16="y16", POINT="point"),
        OBStreamType=SimpleNamespace(COLOR_STREAM="color_stream"),
        OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        get_version=lambda: "2.7.0",
    )


def test_run_preflight_locks_identity_modes_alignment_aggregate_and_stops_pipeline():
    calls: dict[str, object] = {}
    sdk = _fake_sdk(calls)

    report = preflight.run_preflight(
        sdk,
        package_version="2.1.1",
        requested_frames=3,
        sample_stride=1,
        sample_frame=lambda *_args, **_kwargs: _measurement(),
    )

    assert report["status"] == "PASS"
    assert report["serial"] == "CL8326300A0"
    assert report["device"]["name"] == "Femto Bolt"
    assert report["modes"] == {
        "color": {"width": 1920, "height": 1080, "fps": 30, "format": "MJPG"},
        "depth": {"width": 640, "height": 576, "fps": 30, "format": "Y16"},
        "alignment": "software_depth_to_color",
        "frame_aggregate": "FULL_FRAME_REQUIRE",
    }
    assert report["sdk"] == {
        "distribution": "pyorbbecsdk2",
        "distribution_version": "2.1.1",
        "native_version": "2.7.0",
    }
    assert calls["serial_lookup"] == "CL8326300A0"
    assert calls["profile_requests"] == [
        ("color", 1920, 1080, "mjpg", 30),
        ("depth", 640, 576, "y16", 30),
    ]
    assert calls["align_target"] == "color_stream"
    assert calls["point_format"] == "point"
    assert calls["started"].aggregate_mode == "full"
    assert calls["stop_count"] == 1


def test_run_preflight_stops_pipeline_when_sampling_raises():
    calls: dict[str, object] = {}
    sdk = _fake_sdk(calls)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic capture failure")

    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        preflight.run_preflight(
            sdk,
            package_version="2.1.1",
            requested_frames=3,
            sample_frame=fail,
        )

    assert calls["stop_count"] == 1


def test_run_preflight_rejects_wrong_distribution_before_opening_camera():
    calls: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="pyorbbecsdk2==2.1.1"):
        preflight.run_preflight(
            _fake_sdk(calls),
            package_version="2.1.2",
            requested_frames=3,
        )

    assert "pipeline_device" not in calls


def test_canonical_json_is_sorted_finite_and_newline_terminated():
    assert preflight.canonical_json_bytes({"z": 1, "a": [2, 3]}) == b'{"a":[2,3],"z":1}\n'
    with pytest.raises(ValueError):
        preflight.canonical_json_bytes({"bad": float("nan")})


def test_write_report_bundle_never_leaves_pass_marker_for_failure(tmp_path):
    path = tmp_path / "preflight.json"
    marker = tmp_path / "preflight.json.PASS"
    marker.write_text("stale", encoding="utf-8")
    report = {"status": "FAIL", "failures": ["synthetic"]}

    result = preflight.write_report_bundle(path, report)

    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert result["report_sha256"] == preflight.sha256_file(path)
    assert not marker.exists()


def test_write_report_bundle_writes_hash_bound_pass_marker_last(tmp_path):
    path = tmp_path / "preflight.json"
    report = {"status": "PASS", "failures": []}

    result = preflight.write_report_bundle(path, report)

    marker = json.loads((tmp_path / "preflight.json.PASS").read_text(encoding="utf-8"))
    assert marker == {
        "report": "preflight.json",
        "report_sha256": result["report_sha256"],
        "status": "PASS",
    }
