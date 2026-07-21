"""Replay / simulation harness for the streaming tracker.

Two sources, same streaming contract as live:
  - ``make_synthetic_sequence`` — a renderable curved "back" patch under combined camera+patient
    rigid motion and breathing, with **exact ground-truth** 3D points. Used by the tests and
    for accuracy eval (sensors are not on the box yet).
  - ``run_session`` — stream a recorded ``Session`` (RGB+depth) frame-by-frame.

``SyntheticDetector`` plays the role of ``perc_lmh_v4`` without torch (returns the GT pixels +
noise), so the whole fusion/geometry pipeline is testable on any machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import LUNG_ORDER, DetectorResult
from .geometry import CropTransform, Intrinsics, make_T
from .intrinsics import default_intrinsics
from .tracker import StreamingAuscultationTracker, TrackerConfig


# ── synthetic scene ──────────────────────────────────────────────────────────
@dataclass
class SynFrame:
    rgb: np.ndarray
    depth: np.ndarray
    t: float
    depth_scale: float
    gt_xyz_cam: np.ndarray     # (10,3) live skin points (with breathing) [m]
    gt_pixels: np.ndarray      # (10,2) full-frame pixels


def _rot(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _surface(curv: float = 1.2, half: float = 0.12, n: int = 240):
    g = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(g, g)
    gz = curv * (gx ** 2 + gy ** 2)
    obj = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    nrm = np.stack([-2 * curv * gx, -2 * curv * gy, np.ones_like(gx)], -1).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # corner-rich, surface-anchored speckle (each surface point keeps its colour) → stable
    # features for RGB-D odometry, on an otherwise smooth (geometrically textureless) back.
    tex = np.random.RandomState(7).rand(obj.shape[0])
    return obj, nrm, tex


def _gt_object(curv: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """10 auscultation points in object frame, in LUNG_ORDER (R=+x/image-right, L=-x)."""
    rows = np.linspace(-0.08, 0.08, 5)
    col = 0.045
    pts, nrm = [], []
    for r in rows:
        for sign in (+1, -1):          # R then L (matches ORDER R{i},L{i})
            x, y = sign * col, r
            pts.append([x, y, curv * (x ** 2 + y ** 2)])
            n = np.array([-2 * curv * x, -2 * curv * y, 1.0]); n /= np.linalg.norm(n)
            nrm.append(n)
    return np.array(pts), np.array(nrm)


def make_synthetic_sequence(n_frames: int = 120, fps: float = 30.0,
                            intr: Optional[Intrinsics] = None, d0: float = 0.6,
                            motion: bool = True, breathing: bool = True,
                            depth_noise_mm: float = 2.0, background: bool = True,
                            curv: float = 1.2, occlude=None,
                            seed: int = 0) -> Tuple[List[SynFrame], Intrinsics]:
    """``curv`` controls surface curvature (low ≈ flat → exposes ICP tangential sliding).
    ``occlude=(f0,f1,y0frac,y1frac)`` blanks an image band over frames [f0,f1) (kills flow+depth
    there → tests coasting)."""
    rng = np.random.RandomState(seed)
    intr = intr or default_intrinsics(640, 480, hfov_deg=58.0)
    obj, onrm, tex = _surface(curv=curv)
    gt_obj, gt_nrm = _gt_object(curv=curv)
    color = np.stack([tex, 0.4 + 0.4 * tex, 1.0 - tex], -1) * 255.0
    frames: List[SynFrame] = []
    for i in range(n_frames):
        t = i / fps
        if motion:
            ax = np.radians(4.0) * np.sin(2 * np.pi * 0.10 * t)
            ay = np.radians(5.0) * np.sin(2 * np.pi * 0.07 * t + 1.0)
            tx = 0.02 * np.sin(2 * np.pi * 0.05 * t)
            ty = 0.015 * np.sin(2 * np.pi * 0.06 * t + 0.5)
        else:
            ax = ay = tx = ty = 0.0
        R = _rot(ax, ay, 0.0)
        trans = np.array([tx, ty, d0])
        amp = 0.006 * np.sin(2 * np.pi * 0.3 * t) if breathing else 0.0  # 6 mm normal breathing
        cam = (R @ (obj + amp * onrm).T).T + trans
        gcam = (R @ (gt_obj + amp * gt_nrm).T).T + trans
        rgb, depth = _render(cam, color, intr, rng, depth_noise_mm, background)
        if occlude is not None and occlude[0] <= i < occlude[1]:
            H = intr.height
            y0, y1 = int(H * occlude[2]), int(H * occlude[3])
            rgb[y0:y1] = 90               # featureless occluder → LK fails
            depth[y0:y1] = 0              # occluder has no usable depth
        gpix = intr.project(gcam)
        frames.append(SynFrame(rgb, depth, t, intr.depth_scale, gcam, gpix))
    return frames, intr


def _render(cam: np.ndarray, color: np.ndarray, intr: Intrinsics, rng,
            depth_noise_mm: float, background: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    H, W = intr.height, intr.width
    depth_m = np.zeros((H, W), np.float64)
    rgb = np.zeros((H, W, 3), np.uint8)
    if background:                                  # static far wall (the room) → masked out
        depth_m[:] = 1.05
        rgb[:] = (np.random.RandomState(3).rand(H, W, 3) * 255).astype(np.uint8)
    uv = intr.project(cam)
    z = cam[:, 2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 0.05) & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    u, v, z, col = u[ok], v[ok], z[ok], color[ok]
    order = np.argsort(-z)                       # far → near (near overwrites)
    u, v, z, col = u[order], v[order], z[order], col[order]
    for dv in (-1, 0, 1):                          # 3×3 splat to fill the surface
        for du in (-1, 0, 1):
            depth_m[v + dv, u + du] = z
            rgb[v + dv, u + du] = col.astype(np.uint8)
    if depth_noise_mm > 0:
        nz = rng.normal(0, depth_noise_mm / 1000.0, size=depth_m.shape)
        depth_m[depth_m > 0] += nz[depth_m > 0]
    depth = (depth_m * 1000.0 / intr.depth_scale).astype(np.uint16)   # depth_scale mm/unit
    return rgb, depth


# ── synthetic detector (stands in for perc_lmh_v4) ──────────────────────────
class SyntheticDetector:
    """Returns the GT pixels (+ noise) as a detection. Driven by ``set_truth`` each frame."""

    def __init__(self, intr: Intrinsics, res: int = 518, up_bias: float = 0.12,
                 noise_px: float = 1.0, confidence: float = 0.9, seed: int = 1) -> None:
        self.intr = intr
        self.res, self.up_bias = res, up_bias
        self.noise_px, self.confidence = noise_px, confidence
        self.rng = np.random.RandomState(seed)
        self._gt_full: Optional[np.ndarray] = None

    def set_truth(self, gt_pixels_full: np.ndarray) -> None:
        self._gt_full = np.asarray(gt_pixels_full, dtype=np.float64)

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        if self._gt_full is None:
            return None
        H, W = rgb_full.shape[:2]
        crop = CropTransform.center_square(H, W, self.res, self.up_bias)
        det = crop.full_to_det(self._gt_full + self.rng.normal(0, self.noise_px, self._gt_full.shape))
        return DetectorResult(xy_det=det, crop=crop, confidence=self.confidence,
                              part=None, uv=None, order=tuple(LUNG_ORDER))


# ── run harness + metrics ────────────────────────────────────────────────────
def run_over_frames(tracker: StreamingAuscultationTracker, frames: List[SynFrame],
                    detector=None) -> Dict[str, float]:
    errs, per_pt_err = [], {nm: [] for nm in LUNG_ORDER}
    n_tracking, n_total = 0, 0
    for fr in frames:
        if detector is not None and hasattr(detector, "set_truth"):
            detector.set_truth(fr.gt_pixels)
        out = tracker.process(fr.rgb, fr.depth, t=fr.t, depth_scale=fr.depth_scale)
        if not out.initialized:
            continue
        for k, tp in enumerate(out.points):
            n_total += 1
            n_tracking += int(tp.status == "TRACKING")
            if np.isfinite(tp.xyz_camera).all():
                e = float(np.linalg.norm(tp.xyz_camera - fr.gt_xyz_cam[k]))
                errs.append(e); per_pt_err[tp.name].append(e)
    errs = np.array(errs) if errs else np.array([np.inf])
    return {
        "median_err_mm": float(np.median(errs) * 1000.0),
        "p90_err_mm": float(np.percentile(errs, 90) * 1000.0),
        "max_err_mm": float(np.max(errs) * 1000.0),
        "frac_tracking": float(n_tracking / max(n_total, 1)),
        "n_frames": len(frames),
        "n_obs": len(errs),
    }


def run_session(session_dir: str, detector, intr: Intrinsics,
                config: Optional[TrackerConfig] = None, hz: float = 15.0) -> List:
    """Stream a recorded ``Session`` through the tracker (no GT; returns outputs)."""
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from session_io import Session  # type: ignore
    s = Session(session_dir)
    tracker = StreamingAuscultationTracker(detector, intr, config)
    outs = []
    t = s.t0
    while t < s.t0 + s.duration:
        snap = s.aligned_at(t)
        outs.append(tracker.process(snap.rgb, snap.depth, t=t))
        t += 1.0 / hz
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="streaming tracker replay/sim")
    ap.add_argument("--synthetic", action="store_true", help="run the synthetic accuracy sim")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--detect_hz", type=float, default=2.0)
    args = ap.parse_args()
    if args.synthetic:
        frames, intr = make_synthetic_sequence(n_frames=args.frames)
        det = SyntheticDetector(intr)
        cfg = TrackerConfig(detect_interval_s=1.0 / args.detect_hz)
        tracker = StreamingAuscultationTracker(det, intr, cfg)
        m = run_over_frames(tracker, frames, det)
        print("[synthetic]", " ".join(f"{k}={v:.2f}" for k, v in m.items()))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
