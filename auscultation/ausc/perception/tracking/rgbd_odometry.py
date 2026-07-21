"""RGB-D feature odometry — the tangential lock (enabled by aligned RGB-D).

A human back is a smooth, low-geometric-texture surface, so point-to-plane ICP constrains the
surface *normal* well but **slides tangentially** (in-surface) — over a long session the
canonical anchor could drift sideways across the skin. RGB texture (pores, moles, hair) pins
those tangential DOFs, and because the Orbbec RGB and depth are **per-pixel aligned**, every
tracked RGB corner has an *exact* 3D coordinate — so frame↔canonical motion is a clean 3D-3D
problem solved with RANSAC.

``FeatureField`` maintains a set of canonical-frame 3D feature points (born on the patient
surface, anchored once via the current pose) and, each frame, LK-tracks their pixels, reads
their exact depth, and estimates ``T_cam←canon`` by ``ransac_rigid``. Dead features are dropped
and fresh ones seeded, so it runs at constant cost on the live patient region only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import (Intrinsics, invert_T, make_T, ransac_rigid, sample_depth_m,
                       transform_points)


class FeatureField:
    def __init__(self, intr: Intrinsics, max_feats: int = 160, quality: float = 0.01,
                 min_dist: int = 10, reseed_below: int = 80, fb_thresh: float = 1.5,
                 ransac_thresh: float = 0.008, win: int = 21, max_level: int = 3,
                 band: float = 0.25, max_depth_m: float = 3.0) -> None:
        self.intr = intr
        self.max_feats, self.quality, self.min_dist = max_feats, quality, min_dist
        self.reseed_below, self.fb_thresh, self.ransac_thresh = reseed_below, fb_thresh, ransac_thresh
        self.band, self.max_depth_m = band, max_depth_m
        self.pix = np.zeros((0, 2))
        self.canon = np.zeros((0, 3))
        self._prev_gray: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(20260626)
        self._lk = dict(winSize=(win, win), maxLevel=max_level)

    # -- helpers ----------------------------------------------------------
    def _to3d(self, pix: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = sample_depth_m(depth, pix, self.intr.depth_scale, max_m=self.max_depth_m)
        ok = np.isfinite(z)
        cam = np.full((len(pix), 3), np.nan)
        if ok.any():
            cam[ok] = self.intr.unproject(pix[ok], z[ok])
        return cam, ok

    def _body_mask(self, depth: np.ndarray):
        import cv2
        z = depth.astype(np.float64) * self.intr.depth_scale / 1000.0
        H, W = z.shape
        c = z[H // 4:3 * H // 4, W // 4:3 * W // 4]
        v = c[(c > 0.05) & (c < self.max_depth_m)]
        if v.size < 16:
            m = ((z > 0.05) & (z < self.max_depth_m)).astype(np.uint8) * 255
        else:
            d0 = float(np.median(v))
            m = ((np.abs(z - d0) < self.band) & (z > 0.05)).astype(np.uint8) * 255
        return cv2.erode(m, np.ones((self.min_dist, self.min_dist), np.uint8))

    def _seed(self, gray, depth, T_cam_canon: np.ndarray, n: int) -> None:
        import cv2
        if n <= 0:
            return
        mask = self._body_mask(depth)
        for (px, py) in self.pix.astype(int):      # block existing features
            cv2.circle(mask, (int(px), int(py)), self.min_dist, 0, -1)
        feats = cv2.goodFeaturesToTrack(gray, n, self.quality, self.min_dist, mask=mask)
        if feats is None:
            return
        pix = feats.reshape(-1, 2).astype(np.float64)
        cam, ok = self._to3d(pix, depth)
        if not ok.any():
            return
        canon = transform_points(invert_T(T_cam_canon), cam[ok])
        self.pix = np.vstack([self.pix, pix[ok]])
        self.canon = np.vstack([self.canon, canon])

    # -- lifecycle --------------------------------------------------------
    def reset(self, gray, depth, T_cam_canon: np.ndarray) -> None:
        self.pix = np.zeros((0, 2)); self.canon = np.zeros((0, 3))
        self._seed(gray, depth, T_cam_canon, self.max_feats)
        self._prev_gray = gray

    def estimate(self, gray, depth, T_init: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Return (T_cam←canon, inlier_ratio, rmse[m]). Falls back to ``T_init`` if too sparse."""
        import cv2
        if self._prev_gray is None or len(self.pix) < 6 or gray is None:
            if gray is not None:
                self.reset(gray, depth, T_init)
            return T_init, 0.0, float("inf")
        p0 = self.pix.astype(np.float32).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **self._lk)
        p0r, st0, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **self._lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        cur = p1.reshape(-1, 2).astype(np.float64)
        cam, ok = self._to3d(cur, depth)
        alive = (st1.reshape(-1) == 1) & (st0.reshape(-1) == 1) & (fb < self.fb_thresh) & ok
        if alive.sum() < 6:
            self.reset(gray, depth, T_init)
            return T_init, 0.0, float("inf")
        R, t, fit = ransac_rigid(self.canon[alive], cam[alive], thresh=self.ransac_thresh,
                                 rng=self._rng)
        T = make_T(R, t)
        resid = np.linalg.norm((R @ self.canon[alive].T).T + t - cam[alive], axis=1)
        rmse = float(np.median(resid))
        self.pix, self.canon = cur[alive], self.canon[alive]   # keep survivors
        self._prev_gray = gray
        if len(self.pix) < self.reseed_below:                  # top up on the patient surface
            self._seed(gray, depth, T, self.max_feats - len(self.pix))
        return T, fit, rmse
