"""Persist a measured camera exposure-to-host latency calibration.

Measure repeated LED-flash events with the same camera mode used for collection,
then pass the per-event ``host_arrival - exposure`` latencies in milliseconds.
The tool refuses sparse, non-finite, implausible, or high-jitter measurements.

    python ausc/camera_latency_calib.py --backend orbbec --camera-id 0 \
      --width 640 --height 480 --fps 30 --color-format RGB \
      --depth-width 640 --depth-height 576 --depth-fps 30 --depth-format Y16 \
      --align-mode software_depth_to_color --frame-aggregate-mode full_frame_require \
      --samples-ms 12.1 11.8 ... \
      --out camera_latency.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "wam_camera_latency_v1"
MEASUREMENT_METHOD = "hardware_led_flash"
MIN_SAMPLES = 20
MAX_STD_MS = 5.0
MAX_LATENCY_MS = 500.0
CAMERA_TIME_SOURCE_BY_BACKEND = {
    "orbbec": "orbbec_device_and_host_arrival",
    "avfoundation": "opencv_host_arrival",
    "opencv": "opencv_host_arrival",
}
ORBBEC_COLOR_FORMATS = frozenset({"RGB", "BGR", "MJPG", "YUYV", "UYVY", "NV12"})
ORBBEC_ALIGN_MODES = frozenset({"software_depth_to_color", "disabled"})


def validate_depth_mode(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("camera latency capture_mode depth_mode must be a JSON object")
    width, height, fps, pixel_format = (
        value.get("width"), value.get("height"), value.get("fps"), value.get("format"),
    )
    if type(width) is not int or width <= 0:
        raise ValueError("camera latency capture_mode depth_mode width must be positive integer")
    if type(height) is not int or height <= 0:
        raise ValueError("camera latency capture_mode depth_mode height must be positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("camera latency capture_mode depth_mode fps must be finite and positive")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("camera latency capture_mode depth_mode fps must be finite and positive")
    if not isinstance(pixel_format, str) or not pixel_format.strip():
        raise ValueError("camera latency capture_mode depth_mode format must be non-empty")
    return {"width": width, "height": height, "fps": fps, "format": pixel_format.strip()}


def expected_camera_time_source(backend: object) -> str:
    """Return the timestamp semantics implemented by a physical camera backend."""
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("camera latency capture_mode backend must be a non-empty string")
    normalized = backend.strip().lower()
    try:
        return CAMERA_TIME_SOURCE_BY_BACKEND[normalized]
    except KeyError as exc:
        raise ValueError(
            f"camera latency capture_mode backend {normalized!r} is not a physical backend"
        ) from exc


def validate_camera_capture_mode(
    *, backend: object, camera_id: object, width: object, height: object, fps: object,
    time_source: object | None = None, color_format: object = None,
    depth_mode: object = None, align_mode: object = None,
    frame_aggregate_mode: object = None,
) -> dict:
    """Validate the exact camera mode to which a latency measurement applies."""
    expected_time_source = expected_camera_time_source(backend)
    normalized_backend = str(backend).strip().lower()
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera latency capture_mode camera_id must be a non-empty string")
    if type(width) is not int or width <= 0:
        raise ValueError("camera latency capture_mode width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise ValueError("camera latency capture_mode height must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("camera latency capture_mode fps must be a finite positive number")
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("camera latency capture_mode fps must be a finite positive number")
    actual_time_source = expected_time_source if time_source is None else time_source
    if actual_time_source != expected_time_source:
        raise ValueError(
            "camera latency capture_mode time_source must match backend semantics: "
            f"expected {expected_time_source!r}, got {actual_time_source!r}"
        )
    if not isinstance(color_format, str) or not color_format.strip():
        raise ValueError("camera latency capture_mode color_format must be a non-empty string")
    color_format = color_format.strip()
    if normalized_backend == "orbbec":
        if color_format not in ORBBEC_COLOR_FORMATS:
            raise ValueError(
                f"camera latency capture_mode color_format {color_format!r} is unsupported"
            )
        normalized_depth_mode = validate_depth_mode(depth_mode)
        if align_mode not in ORBBEC_ALIGN_MODES:
            raise ValueError(
                f"camera latency capture_mode align_mode must be one of "
                f"{sorted(ORBBEC_ALIGN_MODES)}"
            )
        if frame_aggregate_mode != "full_frame_require":
            raise ValueError(
                "camera latency capture_mode frame_aggregate_mode must be "
                "full_frame_require for Orbbec"
            )
    else:
        if color_format != "opencv_decoded_bgr":
            raise ValueError(
                "camera latency capture_mode color_format for OpenCV must be "
                "'opencv_decoded_bgr'"
            )
        if depth_mode is not None:
            raise ValueError("camera latency capture_mode depth_mode must be null for OpenCV")
        normalized_depth_mode = None
        if align_mode != "disabled":
            raise ValueError("camera latency capture_mode align_mode must be disabled for OpenCV")
        if frame_aggregate_mode != "not_applicable":
            raise ValueError(
                "camera latency capture_mode frame_aggregate_mode must be not_applicable for OpenCV"
            )
    return {
        "backend": normalized_backend,
        "camera_id": camera_id.strip(),
        "width": width,
        "height": height,
        "fps": fps_value,
        "time_source": expected_time_source,
        "color_format": color_format,
        "depth_mode": normalized_depth_mode,
        "align_mode": align_mode,
        "frame_aggregate_mode": frame_aggregate_mode,
    }


def persist_camera_latency(
    path: str | Path,
    samples_ms,  # noqa: ANN001
    *,
    backend: str,
    camera_id: str,
    width: int,
    height: int,
    fps: float,
    color_format: str,
    depth_mode: Mapping[str, object] | None,
    align_mode: str,
    frame_aggregate_mode: str,
    time_source: str | None = None,
) -> dict:
    capture_mode = validate_camera_capture_mode(
        backend=backend, camera_id=camera_id, width=width, height=height, fps=fps,
        time_source=time_source, color_format=color_format, depth_mode=depth_mode,
        align_mode=align_mode, frame_aggregate_mode=frame_aggregate_mode,
    )
    samples = np.asarray(samples_ms, dtype=np.float64)
    if samples.ndim != 1 or samples.size < MIN_SAMPLES:
        raise ValueError(f"camera latency calibration needs at least {MIN_SAMPLES} samples")
    if (not np.isfinite(samples).all() or np.any(samples <= 0)
            or np.any(samples > MAX_LATENCY_MS)):
        raise ValueError("camera latency samples must be finite and within (0, 500] ms")
    latency_ms = float(np.mean(samples))
    latency_std_ms = float(np.std(samples))
    if latency_std_ms > MAX_STD_MS:
        raise ValueError(
            f"camera latency standard deviation {latency_std_ms:.3f} ms exceeds {MAX_STD_MS:.1f} ms"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "latency_definition": "host_arrival_minus_exposure",
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "n_samples": int(samples.size),
        "measurement_method": MEASUREMENT_METHOD,
        "capture_mode": capture_mode,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return payload


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-ms", type=float, nargs="+", required=True,
                        help="at least 20 LED-flash host-arrival-minus-exposure values in ms")
    parser.add_argument("--backend", required=True,
                        help="exact collection backend used while measuring")
    parser.add_argument("--camera-id", required=True,
                        help="stable camera serial/id/index used while measuring")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--color-format", required=True,
                        help="Orbbec SDK color format (for example RGB), or opencv_decoded_bgr")
    parser.add_argument("--depth-width", type=int)
    parser.add_argument("--depth-height", type=int)
    parser.add_argument("--depth-fps", type=float)
    parser.add_argument("--depth-format")
    parser.add_argument("--align-mode", required=True,
                        choices=["software_depth_to_color", "disabled"])
    parser.add_argument("--frame-aggregate-mode", required=True,
                        choices=["full_frame_require", "not_applicable"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    depth_values = (args.depth_width, args.depth_height, args.depth_fps, args.depth_format)
    depth_mode = (None if all(value is None for value in depth_values) else {
        "width": args.depth_width, "height": args.depth_height,
        "fps": args.depth_fps, "format": args.depth_format,
    })
    payload = persist_camera_latency(
        args.out, args.samples_ms, backend=args.backend, camera_id=args.camera_id,
        width=args.width, height=args.height, fps=args.fps,
        color_format=args.color_format, depth_mode=depth_mode, align_mode=args.align_mode,
        frame_aggregate_mode=args.frame_aggregate_mode,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
