"""Fail-closed Orbbec RGB-D geometry preflight for Lingbox hand-eye calibration.

This tool does not calibrate the robot.  It commissions the exact Femto Bolt
RGB-D geometry path used by calibration: software depth-to-colour alignment is
compared, pixel for pixel, with Orbbec's own POINT point-cloud output from the
same aligned frame set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


EXPECTED_SERIAL = "CL8326300A0"
EXPECTED_DISTRIBUTION = "pyorbbecsdk2"
EXPECTED_DISTRIBUTION_VERSION = "2.1.1"
DISTORTION_ORDER = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
MIN_FRAMES = 3
MIN_VALID_SAMPLES_PER_FRAME = 100
MIN_X_COVERAGE = 0.60
MIN_Y_COVERAGE = 0.60
MAX_P95_ERROR_MM = 2.0
MAX_SINGLE_ERROR_MM = 5.0
DEFAULT_SAMPLE_STRIDE = 8

EXPECTED_MODES = {
    "color": {"width": 1920, "height": 1080, "fps": 30, "format": "MJPG"},
    "depth": {"width": 640, "height": 576, "fps": 30, "format": "Y16"},
    "alignment": "software_depth_to_color",
    "frame_aggregate": "FULL_FRAME_REQUIRE",
}


@dataclass
class FrameGeometryEvidence:
    """Comparison evidence for one aligned frame set."""

    errors_mm: np.ndarray
    valid_sample_count: int
    candidate_sample_count: int
    nonfinite_correspondence_count: int
    invalid_sdk_correspondence_count: int
    x_coverage_fraction: float
    y_coverage_fraction: float
    width: int
    height: int
    sample_stride: int
    depth_scale_mm: float
    sdk_position_scale_mm: float


def _finite_positive(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _coverage(coordinates: np.ndarray, extent: int) -> float:
    if coordinates.size == 0 or extent <= 1:
        return 0.0
    return float((coordinates.max() - coordinates.min()) / float(extent - 1))


def compare_aligned_depth_to_sdk_xyz(
    depth_raw: np.ndarray,
    *,
    depth_scale_mm: float,
    sdk_xyz_raw: np.ndarray,
    sdk_position_scale_mm: float,
    intrinsics: dict[str, float | int],
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
) -> FrameGeometryEvidence:
    """Compare ``K^-1 * aligned_depth`` with SDK XYZ in millimetres.

    Invalid depth (zero) is not a candidate.  A non-finite or non-positive SDK
    Z at a positive-depth candidate is retained as explicit failing evidence;
    it is never silently dropped from the gate.
    """

    depth = np.asarray(depth_raw)
    if depth.ndim != 2:
        raise ValueError("aligned depth must be a two-dimensional array")
    height, width = depth.shape
    xyz_raw = np.asarray(sdk_xyz_raw)
    if xyz_raw.shape != (height, width, 3):
        raise ValueError(
            f"SDK XYZ shape {xyz_raw.shape!r} does not match aligned depth {(height, width)!r}"
        )
    if int(intrinsics.get("width", -1)) != width or int(intrinsics.get("height", -1)) != height:
        raise ValueError("intrinsics resolution does not match aligned depth")
    if isinstance(sample_stride, bool) or int(sample_stride) != sample_stride or sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer")
    stride = int(sample_stride)
    depth_scale = _finite_positive(depth_scale_mm, "depth_scale_mm")
    point_scale = _finite_positive(sdk_position_scale_mm, "sdk_position_scale_mm")
    fx = _finite_positive(intrinsics.get("fx"), "intrinsics.fx")
    fy = _finite_positive(intrinsics.get("fy"), "intrinsics.fy")
    try:
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("intrinsics principal point is invalid") from exc
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("intrinsics principal point is invalid")

    y_grid, x_grid = np.indices((height, width), dtype=np.float64)
    sampled = np.zeros((height, width), dtype=bool)
    sampled[::stride, ::stride] = True
    depth_mm = depth.astype(np.float64, copy=False) * depth_scale
    candidate = sampled & np.isfinite(depth_mm) & (depth_mm > 0)

    sdk_xyz_mm = xyz_raw.astype(np.float64, copy=False) * point_scale
    sdk_finite = np.all(np.isfinite(sdk_xyz_mm), axis=-1)
    sdk_positive_z = sdk_xyz_mm[..., 2] > 0
    nonfinite = candidate & ~sdk_finite
    invalid_sdk = candidate & sdk_finite & ~sdk_positive_z
    valid = candidate & sdk_finite & sdk_positive_z

    z = depth_mm[valid]
    x_pixels = x_grid[valid]
    y_pixels = y_grid[valid]
    expected = np.column_stack(
        (
            (x_pixels - cx) * z / fx,
            (y_pixels - cy) * z / fy,
            z,
        )
    )
    observed = sdk_xyz_mm[valid]
    errors = np.linalg.norm(observed - expected, axis=1)

    return FrameGeometryEvidence(
        errors_mm=np.asarray(errors, dtype=np.float64),
        valid_sample_count=int(np.count_nonzero(valid)),
        candidate_sample_count=int(np.count_nonzero(candidate)),
        nonfinite_correspondence_count=int(np.count_nonzero(nonfinite)),
        invalid_sdk_correspondence_count=int(np.count_nonzero(invalid_sdk)),
        x_coverage_fraction=_coverage(x_pixels, width),
        y_coverage_fraction=_coverage(y_pixels, height),
        width=width,
        height=height,
        sample_stride=stride,
        depth_scale_mm=depth_scale,
        sdk_position_scale_mm=point_scale,
    )


def evaluate_geometry_gate(
    frames: Sequence[FrameGeometryEvidence],
) -> dict[str, object]:
    """Apply the commissioning thresholds without consulting camera hardware."""

    evidence = list(frames)
    frame_count = len(evidence)
    valid_counts = [frame.valid_sample_count for frame in evidence]
    valid_min = min(valid_counts, default=0)
    valid_total = sum(valid_counts)
    candidate_total = sum(frame.candidate_sample_count for frame in evidence)
    nonfinite_count = sum(frame.nonfinite_correspondence_count for frame in evidence)
    invalid_sdk_count = sum(frame.invalid_sdk_correspondence_count for frame in evidence)
    x_coverage = min((frame.x_coverage_fraction for frame in evidence), default=0.0)
    y_coverage = min((frame.y_coverage_fraction for frame in evidence), default=0.0)
    error_arrays = [frame.errors_mm for frame in evidence if frame.errors_mm.size]
    all_errors = np.concatenate(error_arrays) if error_arrays else np.empty(0, dtype=np.float64)
    nonfinite_error_count = int(np.count_nonzero(~np.isfinite(all_errors)))
    finite_errors = all_errors[np.isfinite(all_errors)]
    p95 = float(np.percentile(finite_errors, 95)) if finite_errors.size else None
    maximum = float(np.max(finite_errors)) if finite_errors.size else None

    failures: list[str] = []
    if frame_count < MIN_FRAMES:
        failures.append("frames_below_3")
    if valid_min < MIN_VALID_SAMPLES_PER_FRAME:
        failures.append("valid_samples_below_100")
    if nonfinite_count or nonfinite_error_count:
        failures.append("nonfinite_correspondences")
    if invalid_sdk_count:
        failures.append("invalid_sdk_correspondences")
    if x_coverage < MIN_X_COVERAGE:
        failures.append("x_coverage_below_0.6")
    if y_coverage < MIN_Y_COVERAGE:
        failures.append("y_coverage_below_0.6")
    if p95 is None or p95 > MAX_P95_ERROR_MM:
        failures.append("p95_error_above_2mm")
    if maximum is None or maximum > MAX_SINGLE_ERROR_MM:
        failures.append("max_error_above_5mm")

    return {
        "frames": {"count": frame_count, "minimum_required": MIN_FRAMES},
        "samples": {
            "candidate_total": candidate_total,
            "valid_total": valid_total,
            "valid_min_per_frame": valid_min,
            "minimum_valid_per_frame_required": MIN_VALID_SAMPLES_PER_FRAME,
            "nonfinite_correspondences": nonfinite_count + nonfinite_error_count,
            "invalid_sdk_correspondences": invalid_sdk_count,
        },
        "coverage": {
            "x_fraction_min": float(x_coverage),
            "y_fraction_min": float(y_coverage),
            "x_minimum_required": MIN_X_COVERAGE,
            "y_minimum_required": MIN_Y_COVERAGE,
        },
        "errors_mm": {
            "p95": p95,
            "max": maximum,
            "p95_limit": MAX_P95_ERROR_MM,
            "max_limit": MAX_SINGLE_ERROR_MM,
        },
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def extract_active_profile_calibration(active_video_profile) -> dict[str, object]:  # noqa: ANN001
    """Read calibration only from the exact profile selected for activation."""

    try:
        width = int(active_video_profile.get_width())
        height = int(active_video_profile.get_height())
        intrinsic = active_video_profile.get_intrinsic()
        distortion = active_video_profile.get_distortion()
        intrinsics = {
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
            "width": width,
            "height": height,
        }
        coefficients = [float(getattr(distortion, name)) for name in DISTORTION_ORDER]
    except Exception as exc:  # SDK exception type varies by wheel/platform
        raise RuntimeError("active Orbbec profile calibration is unavailable") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("active Orbbec profile resolution is invalid")
    if (
        not all(math.isfinite(float(intrinsics[key])) for key in ("fx", "fy", "cx", "cy"))
        or float(intrinsics["fx"]) <= 0
        or float(intrinsics["fy"]) <= 0
        or not all(math.isfinite(value) for value in coefficients)
    ):
        raise RuntimeError("active Orbbec profile calibration is non-finite or invalid")
    return {
        "K": [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "intrinsics": intrinsics,
        "distortion": {
            "model": "model_api_unexposed",
            "coefficient_order": list(DISTORTION_ORDER),
            "coefficients": coefficients,
        },
    }


def _as_frameset(frame):  # noqa: ANN001
    if frame is None:
        return None
    if hasattr(frame, "get_depth_frame") and hasattr(frame, "get_color_frame"):
        return frame
    if hasattr(frame, "as_frame_set"):
        frameset = frame.as_frame_set()
        if (
            frameset is not None
            and hasattr(frameset, "get_depth_frame")
            and hasattr(frameset, "get_color_frame")
        ):
            return frameset
    return None


def _buffer_array(data, dtype: np.dtype, shape: tuple[int, ...], label: str) -> np.ndarray:  # noqa: ANN001
    array = np.frombuffer(data, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(f"{label} buffer has {array.size} values; expected {expected}")
    return np.ascontiguousarray(array.reshape(shape)).copy()


def sample_one_aligned_frame(
    pipeline,
    align_filter,
    point_cloud_filter,
    intrinsics: dict[str, float | int],
    *,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
) -> FrameGeometryEvidence:
    """Capture and compare one frameset, preserving aligned-frame identity."""

    raw_frameset = pipeline.wait_for_frames(1000)
    if raw_frameset is None:
        raise RuntimeError("Orbbec frame timeout")
    aligned_output = align_filter.process(raw_frameset)
    aligned_frameset = _as_frameset(aligned_output)
    if aligned_frameset is None:
        raise RuntimeError("AlignFilter did not return an RGB-D frameset")
    depth_frame = aligned_frameset.get_depth_frame()
    color_frame = aligned_frameset.get_color_frame()
    if depth_frame is None or color_frame is None:
        raise RuntimeError("aligned frameset is missing color or depth")

    # This exact object is also the source of aligned depth above.  Do not call
    # wait_for_frames again or synthesize a second frameset for point-cloud XYZ.
    point_cloud_frame = point_cloud_filter.process(aligned_frameset)
    if point_cloud_frame is None:
        raise RuntimeError("PointCloudFilter returned no POINT frame")
    points = (
        point_cloud_frame.as_points_frame()
        if hasattr(point_cloud_frame, "as_points_frame")
        else point_cloud_frame
    )
    depth_width = int(depth_frame.get_width())
    depth_height = int(depth_frame.get_height())
    point_width = int(points.get_width())
    point_height = int(points.get_height())
    if (point_width, point_height) != (depth_width, depth_height):
        raise RuntimeError("SDK point cloud resolution does not match aligned depth")
    depth = _buffer_array(
        depth_frame.get_data(), np.dtype(np.uint16), (depth_height, depth_width), "depth"
    )
    sdk_xyz = _buffer_array(
        points.get_data(), np.dtype(np.float32), (point_height, point_width, 3), "POINT"
    )
    return compare_aligned_depth_to_sdk_xyz(
        depth,
        depth_scale_mm=float(depth_frame.get_depth_scale()),
        sdk_xyz_raw=sdk_xyz,
        sdk_position_scale_mm=float(points.get_position_value_scale()),
        intrinsics=intrinsics,
        sample_stride=sample_stride,
    )


def _format_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value).rsplit(".", 1)[-1].upper()


def _profile_mode(profile) -> dict[str, int | str]:  # noqa: ANN001
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": _format_name(profile.get_format()),
    }


def _require_exact_profile(profile, expected: dict[str, int | str], label: str):  # noqa: ANN001
    video_profile = profile.as_video_stream_profile()
    actual = _profile_mode(video_profile)
    if actual != expected:
        raise RuntimeError(f"active Orbbec {label} profile mismatch: {actual!r}")
    return video_profile


def _frame_summary(frame: FrameGeometryEvidence) -> dict[str, object]:
    finite_errors = frame.errors_mm[np.isfinite(frame.errors_mm)]
    return {
        "candidate_samples": frame.candidate_sample_count,
        "valid_samples": frame.valid_sample_count,
        "nonfinite_correspondences": frame.nonfinite_correspondence_count,
        "invalid_sdk_correspondences": frame.invalid_sdk_correspondence_count,
        "x_coverage_fraction": frame.x_coverage_fraction,
        "y_coverage_fraction": frame.y_coverage_fraction,
        "error_p95_mm": float(np.percentile(finite_errors, 95)) if finite_errors.size else None,
        "error_max_mm": float(np.max(finite_errors)) if finite_errors.size else None,
        "aligned_resolution": {"width": frame.width, "height": frame.height},
        "sample_stride": frame.sample_stride,
        "depth_scale_mm": frame.depth_scale_mm,
        "sdk_position_scale_mm": frame.sdk_position_scale_mm,
    }


def run_preflight(
    sdk,
    *,
    package_version: str | None = None,
    requested_frames: int = MIN_FRAMES,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
    sample_frame: Callable[..., FrameGeometryEvidence] = sample_one_aligned_frame,
) -> dict[str, object]:
    """Open the exact Femto Bolt once, acquire evidence, and always stop it."""

    observed_package_version = (
        package_version
        if package_version is not None
        else importlib.metadata.version(EXPECTED_DISTRIBUTION)
    )
    if observed_package_version != EXPECTED_DISTRIBUTION_VERSION:
        raise RuntimeError(
            f"requires {EXPECTED_DISTRIBUTION}=={EXPECTED_DISTRIBUTION_VERSION}; "
            f"found {observed_package_version}"
        )
    if isinstance(requested_frames, bool) or int(requested_frames) != requested_frames:
        raise ValueError("requested_frames must be an integer")
    if int(requested_frames) < MIN_FRAMES:
        raise ValueError(f"requested_frames must be at least {MIN_FRAMES}")
    if isinstance(sample_stride, bool) or int(sample_stride) != sample_stride or sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer")

    pipeline = None
    try:
        context = sdk.Context()
        devices = context.query_devices()
        device = devices.get_device_by_serial_number(EXPECTED_SERIAL)
        info = device.get_device_info()
        actual_serial = str(info.get_serial_number())
        device_name = str(info.get_name())
        if actual_serial != EXPECTED_SERIAL:
            raise RuntimeError(
                f"Orbbec serial mismatch: expected {EXPECTED_SERIAL}, got {actual_serial}"
            )
        if "femto bolt" not in " ".join(device_name.lower().split()):
            raise RuntimeError(f"serial {actual_serial} is not identified as a Femto Bolt")

        pipeline = sdk.Pipeline(device)
        config = sdk.Config()
        color_profile = pipeline.get_stream_profile_list(
            sdk.OBSensorType.COLOR_SENSOR
        ).get_video_stream_profile(1920, 1080, sdk.OBFormat.MJPG, 30)
        depth_profile = pipeline.get_stream_profile_list(
            sdk.OBSensorType.DEPTH_SENSOR
        ).get_video_stream_profile(640, 576, sdk.OBFormat.Y16, 30)
        active_color_profile = _require_exact_profile(
            color_profile, EXPECTED_MODES["color"], "color"
        )
        _require_exact_profile(depth_profile, EXPECTED_MODES["depth"], "depth")
        calibration = extract_active_profile_calibration(active_color_profile)

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
        point_cloud_filter = sdk.PointCloudFilter()
        point_cloud_filter.set_create_point_format(sdk.OBFormat.POINT)
        pipeline.start(config)

        frames = [
            sample_frame(
                pipeline,
                align_filter,
                point_cloud_filter,
                calibration["intrinsics"],
                sample_stride=int(sample_stride),
            )
            for _ in range(int(requested_frames))
        ]
        gate = evaluate_geometry_gate(frames)
        return {
            "schema": "wam_handeye_orbbec_preflight_v1",
            "serial": actual_serial,
            "device": {
                "name": device_name,
                "serial": actual_serial,
                "firmware_version": str(info.get_firmware_version()),
            },
            "modes": json.loads(json.dumps(EXPECTED_MODES)),
            "K": calibration["K"],
            "intrinsics": calibration["intrinsics"],
            "distortion": calibration["distortion"],
            "sdk": {
                "distribution": EXPECTED_DISTRIBUTION,
                "distribution_version": observed_package_version,
                "native_version": str(sdk.get_version()),
            },
            "comparison": {
                "depth_xyz": "aligned_depth_K_inverse_times_z",
                "sdk_xyz": "PointCloudFilter_POINT_same_aligned_frameset",
                "units": "millimetres",
            },
            "frames": gate["frames"],
            "frame_evidence": [_frame_summary(frame) for frame in frames],
            "sampling": {"stride_pixels": int(sample_stride), **gate["samples"]},
            "coverage": gate["coverage"],
            "errors_mm": gate["errors_mm"],
            "failures": gate["failures"],
            "status": gate["status"],
        }
    finally:
        if pipeline is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                pipeline.stop()
            except Exception:
                if not active_exception:
                    raise


def canonical_json_bytes(value: object) -> bytes:
    """Return stable, finite JSON bytes suitable for hashing."""

    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while persisting preflight evidence")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_report_bundle(path: Path | str, report: dict[str, object]) -> dict[str, str]:
    """Write report first and a hash-bound PASS marker only after success."""

    report_path = Path(path)
    pass_path = report_path.with_name(report_path.name + ".PASS")
    try:
        pass_path.unlink()
    except FileNotFoundError:
        pass
    _atomic_write(report_path, canonical_json_bytes(report))
    report_sha256 = sha256_file(report_path)
    if report.get("status") == "PASS":
        marker = {
            "report": report_path.name,
            "report_sha256": report_sha256,
            "status": "PASS",
        }
        _atomic_write(pass_path, canonical_json_bytes(marker))
    return {"report_sha256": report_sha256, "report_path": str(report_path)}


def _failure_report(exc: Exception, package_version: str | None) -> dict[str, object]:
    return {
        "schema": "wam_handeye_orbbec_preflight_v1",
        "serial": EXPECTED_SERIAL,
        "device": {"name": None, "serial": EXPECTED_SERIAL, "firmware_version": None},
        "modes": json.loads(json.dumps(EXPECTED_MODES)),
        "K": None,
        "intrinsics": None,
        "distortion": {
            "model": "model_api_unexposed",
            "coefficient_order": list(DISTORTION_ORDER),
            "coefficients": None,
        },
        "sdk": {
            "distribution": EXPECTED_DISTRIBUTION,
            "distribution_version": package_version,
            "native_version": None,
        },
        "comparison": {
            "depth_xyz": "aligned_depth_K_inverse_times_z",
            "sdk_xyz": "PointCloudFilter_POINT_same_aligned_frameset",
            "units": "millimetres",
        },
        "frames": {"count": 0, "minimum_required": MIN_FRAMES},
        "frame_evidence": [],
        "sampling": {
            "stride_pixels": DEFAULT_SAMPLE_STRIDE,
            "candidate_total": 0,
            "valid_total": 0,
            "valid_min_per_frame": 0,
            "minimum_valid_per_frame_required": MIN_VALID_SAMPLES_PER_FRAME,
            "nonfinite_correspondences": 0,
            "invalid_sdk_correspondences": 0,
        },
        "coverage": {
            "x_fraction_min": 0.0,
            "y_fraction_min": 0.0,
            "x_minimum_required": MIN_X_COVERAGE,
            "y_minimum_required": MIN_Y_COVERAGE,
        },
        "errors_mm": {
            "p95": None,
            "max": None,
            "p95_limit": MAX_P95_ERROR_MM,
            "max_limit": MAX_SINGLE_ERROR_MM,
        },
        "failures": [f"{type(exc).__name__}: {exc}"],
        "status": "FAIL",
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commission exact Femto Bolt aligned-depth geometry before hand-eye calibration."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("handeye_orbbec_preflight.json"),
        help="Canonical JSON report path; a .PASS sibling is written only on success.",
    )
    parser.add_argument("--frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--sample-stride", type=int, default=DEFAULT_SAMPLE_STRIDE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    package_version = None
    try:
        package_version = importlib.metadata.version(EXPECTED_DISTRIBUTION)
        import pyorbbecsdk as sdk

        report = run_preflight(
            sdk,
            package_version=package_version,
            requested_frames=args.frames,
            sample_stride=args.sample_stride,
        )
    except Exception as exc:
        report = _failure_report(exc, package_version)
    try:
        written = write_report_bundle(args.report, report)
    except Exception as exc:
        print(f"failed to write preflight report: {exc}", file=sys.stderr)
        return 3
    print(canonical_json_bytes({"report": written, "status": report["status"]}).decode().strip())
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
