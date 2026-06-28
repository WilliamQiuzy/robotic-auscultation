"""Short-term 2D tracking channel (the **F** evidence source).

Bridges the gap between (throttled) detector calls: it tracks the 10 point pixels from the
previous frame to the current one so the 3D estimate updates every frame and survives short
detector dropouts. Default backend is pyramidal **Lucas–Kanade** optical flow with a
**forward–backward** consistency check (cv2 only, CPU, real-time). A ``CoTracker3`` backend
hook is provided for the GPU upgrade path (better through occlusion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class FlowResult:
    pts: np.ndarray         # (N,2) new full-frame pixels (NaN where lost)
    valid: np.ndarray       # (N,) bool — forward-backward consistent
    fb_err: np.ndarray      # (N,) forward-backward error [px] (inf where lost)


class ShortTermTracker(Protocol):
    def track(self, prev_gray: np.ndarray, cur_gray: np.ndarray,
              prev_pts: np.ndarray) -> FlowResult:
        ...


class LKFlow:
    """Pyramidal Lucas–Kanade with a forward–backward consistency gate."""

    def __init__(self, win: int = 21, max_level: int = 3, fb_thresh: float = 2.0) -> None:
        import cv2
        self._cv2 = cv2
        self.fb_thresh = fb_thresh
        self.params = dict(
            winSize=(win, win), maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @staticmethod
    def to_gray(rgb: np.ndarray) -> np.ndarray:
        import cv2
        if rgb.ndim == 2:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def track(self, prev_gray: np.ndarray, cur_gray: np.ndarray,
              prev_pts: np.ndarray) -> FlowResult:
        cv2 = self._cv2
        p0 = np.asarray(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        n = len(p0)
        if n == 0:                                       # cv2.calcOpticalFlowPyrLK asserts on 0 points
            return FlowResult(pts=np.zeros((0, 2)), valid=np.zeros(0, bool), fb_err=np.zeros(0))
        ok_in = np.isfinite(p0).all(axis=(1, 2))
        p0_safe = np.nan_to_num(p0)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0_safe, None, **self.params)
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **self.params)
        fb = np.linalg.norm((p0_safe - p0r).reshape(n, 2), axis=1)
        st = (st1.reshape(n) == 1) & (st0.reshape(n) == 1) & ok_in
        valid = st & (fb < self.fb_thresh)
        out = p1.reshape(n, 2).astype(np.float64)
        out[~valid] = np.nan
        fb_err = np.where(valid, fb, np.inf)
        return FlowResult(pts=out, valid=valid, fb_err=fb_err)


class CoTrackerFlow:
    """Optional GPU backend (CoTracker3 online). Lazily imported; raises if unavailable.

    Kept as a thin hook so the orchestrator can opt in via ``flow_backend="cotracker"`` when a
    GPU + the model are present. Implementation deferred to the upgrade path; LK is the default.
    """

    def __init__(self, *_, **__):  # pragma: no cover - upgrade path
        raise NotImplementedError(
            "CoTracker3 backend not wired yet; use LKFlow (default). "
            "Upgrade path: facebookresearch/co-tracker online predictor.")
