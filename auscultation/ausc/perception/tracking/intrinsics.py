"""Camera intrinsics acquisition & IO for the tracker.

``CameraStream`` does not expose intrinsics, so the tracker needs them from one of:
  - the Orbbec SDK pipeline (``orbbec_intrinsics``) — the depth-aligned colour profile,
  - a saved calibration JSON (``load_intrinsics`` / ``save_intrinsics``), or
  - an FOV-based approximation (``default_intrinsics``) for bring-up / synthetic data.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .geometry import Intrinsics


def orbbec_intrinsics(pipeline, depth_scale: float = 1.0) -> Optional[Intrinsics]:
    """Read colour-stream intrinsics from a started pyorbbecsdk ``Pipeline``.

    Depth is aligned to colour upstream (``CameraStream(align_depth_to_color=True)``), so the
    colour intrinsics apply to the aligned depth too. Returns None if the SDK call is missing.
    """
    try:
        import pyorbbecsdk as ob  # noqa: F401
        prof = (pipeline.get_stream_profile_list(__import__("pyorbbecsdk").OBSensorType.COLOR_SENSOR)
                .get_default_video_stream_profile())
        vsp = prof.as_video_stream_profile()
        it = vsp.get_intrinsic()
        return Intrinsics(fx=float(it.fx), fy=float(it.fy), cx=float(it.cx), cy=float(it.cy),
                          width=int(vsp.get_width()), height=int(vsp.get_height()),
                          depth_scale=float(depth_scale))
    except Exception:  # noqa: BLE001 - SDK build/profile variations
        return None


def default_intrinsics(width: int, height: int, hfov_deg: float = 60.0,
                       depth_scale: float = 1.0) -> Intrinsics:
    """FOV-based pinhole approximation (square pixels, centred principal point)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                      width=width, height=height, depth_scale=depth_scale)


def save_intrinsics(path: str, intr: Intrinsics) -> None:
    with open(path, "w") as f:
        json.dump({"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                   "width": intr.width, "height": intr.height,
                   "depth_scale": intr.depth_scale}, f, indent=2)


def load_intrinsics(path: str) -> Intrinsics:
    with open(path) as f:
        d = json.load(f)
    return Intrinsics(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                      width=d["width"], height=d["height"],
                      depth_scale=d.get("depth_scale", 1.0))
