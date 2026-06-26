"""Geometric primitives for the streaming auscultation tracker.

Pure numpy/scipy (no torch, no hardware) so the whole tracking core is testable on any
machine. Everything works in **metres** in the camera frame (depth is converted from raw
uint16 + ``depth_scale`` mm-per-unit on the way in).

Contents:
  - ``Intrinsics``    pinhole model for the (depth-aligned) colour stream + unproject/project
  - ``CropTransform`` maps detector space (res×res, after ``center_square``) ↔ full-frame px
  - depth helpers     ``sample_depth_m``, ``pointmap_from_depth``, ``normals_from_pointmap``
  - SE(3) helpers     ``make_T``, ``invert_T``, ``transform_points``
  - registration      ``rigid_align`` (Kabsch/Umeyama), ``point_to_plane_icp``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # cKDTree makes ICP association fast; degrade to brute force if scipy is absent.
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # noqa: BLE001
    cKDTree = None  # type: ignore


# ── camera model ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for the colour stream (depth is aligned to colour upstream)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1.0  # mm per raw depth unit (Orbbec frame metadata)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def unproject(self, uv: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        """(N,2) pixels + (N,) depth[m] → (N,3) camera-frame points [m]."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        z = np.asarray(z_m, dtype=np.float64).reshape(-1)
        x = (uv[:, 0] - self.cx) * z / self.fx
        y = (uv[:, 1] - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame points [m] → (N,2) pixels (z>0 assumed)."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        z = np.clip(xyz[:, 2], 1e-6, None)
        u = xyz[:, 0] * self.fx / z + self.cx
        v = xyz[:, 1] * self.fy / z + self.cy
        return np.stack([u, v], axis=1)

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        """Intrinsics for an image resized by (sx, sy)."""
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                          int(round(self.width * sx)), int(round(self.height * sy)),
                          self.depth_scale)


@dataclass(frozen=True)
class CropTransform:
    """Affine between detector space (res×res square crop) and full-frame pixels.

    Mirrors ``infer_overlay.center_square``: a side-``side`` square at (x0,y0) resized to res.
    """

    x0: int
    y0: int
    side: int
    res: int

    @staticmethod
    def center_square(h: int, w: int, res: int, up_bias: float = 0.12) -> "CropTransform":
        s = min(h, w)
        x0 = (w - s) // 2
        y0 = int((h - s) * (0.5 - up_bias))
        y0 = max(0, min(h - s, y0))
        return CropTransform(x0=x0, y0=y0, side=s, res=res)

    def det_to_full(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) detector-space xy → (N,2) full-frame pixels."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        scale = self.side / float(self.res)
        return np.stack([self.x0 + xy[:, 0] * scale, self.y0 + xy[:, 1] * scale], axis=1)

    def full_to_det(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        scale = self.res / float(self.side)
        return np.stack([(uv[:, 0] - self.x0) * scale, (uv[:, 1] - self.y0) * scale], axis=1)


# ── depth helpers ───────────────────────────────────────────────────────────
def sample_depth_m(depth: np.ndarray, uv: np.ndarray, depth_scale: float,
                   win: int = 2, max_m: float = 4.0) -> np.ndarray:
    """Robust depth [m] at pixels ``uv`` = median of valid depths in a ±win window.

    Returns NaN where no valid depth is found (so callers can drop the measurement).
    """
    H, W = depth.shape
    uv = np.atleast_2d(np.asarray(uv))
    out = np.full(len(uv), np.nan, dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        y0, y1 = max(0, vi - win), min(H, vi + win + 1)
        x0, x1 = max(0, ui - win), min(W, ui + win + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float64) * depth_scale / 1000.0  # → metres
        valid = patch[(patch > 0.05) & (patch < max_m)]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def pointmap_from_depth(depth: np.ndarray, intr: Intrinsics, stride: int = 1,
                        max_m: float = 4.0) -> np.ndarray:
    """Organized (H',W',3) camera-frame point map [m]; invalid pixels → NaN."""
    H, W = depth.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    z = depth[np.ix_(ys, xs)].astype(np.float64) * intr.depth_scale / 1000.0
    uu, vv = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    x = (uu - intr.cx) * z / intr.fx
    y = (vv - intr.cy) * z / intr.fy
    P = np.stack([x, y, z], axis=-1)
    P[(z <= 0.05) | (z >= max_m)] = np.nan
    return P


def normals_from_pointmap(P: np.ndarray) -> np.ndarray:
    """Per-pixel unit normals from an organized point map via neighbour cross products."""
    H, W, _ = P.shape
    n = np.full_like(P, np.nan)
    dx = P[1:-1, 2:, :] - P[1:-1, :-2, :]
    dy = P[2:, 1:-1, :] - P[:-2, 1:-1, :]
    cr = np.cross(dx, dy)
    norm = np.linalg.norm(cr, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = cr / norm
    # flip to face the camera (-Z)
    flip = cr[..., 2] > 0
    cr[flip] = -cr[flip]
    n[1:-1, 1:-1, :] = cr
    return n


def valid_points(P: np.ndarray, normals: Optional[np.ndarray] = None):
    """Flatten an organized point map to (M,3) valid points (+ matching normals)."""
    flat = P.reshape(-1, 3)
    ok = np.isfinite(flat).all(axis=1)
    if normals is not None:
        nfl = normals.reshape(-1, 3)
        ok = ok & np.isfinite(nfl).all(axis=1)
        return flat[ok], nfl[ok]
    return flat[ok], None


def body_band_mask(P: np.ndarray, central_frac: float = 0.5, band: float = 0.25) -> np.ndarray:
    """Boolean (H,W) mask of the patient surface: a depth band around the median depth of the
    central image region. The patient fills the central view; the bed/background sits outside
    the band. Aligned RGB-D makes this a free, reliable foreground segmentation for ICP/surfels.
    """
    H, W, _ = P.shape
    z = P[..., 2]
    y0, y1 = int(H * (1 - central_frac) / 2), int(H * (1 + central_frac) / 2)
    x0, x1 = int(W * (1 - central_frac) / 2), int(W * (1 + central_frac) / 2)
    central = z[y0:y1, x0:x1]
    valid = central[np.isfinite(central) & (central > 0)]
    if valid.size < 16:
        return np.isfinite(z) & (z > 0)
    d0 = float(np.median(valid))
    return np.isfinite(z) & (np.abs(z - d0) < band)


def axial_noise_var(z_m: np.ndarray, k: float = 0.0025, floor_mm: float = 1.5) -> np.ndarray:
    """Depth-camera axial noise variance [m^2]: σ ≈ max(floor, k·z²) (Orbbec/structured-light
    noise grows ~quadratically with range). Used to weight EKF measurements by distance."""
    z = np.asarray(z_m, dtype=np.float64)
    sigma = np.maximum(floor_mm / 1000.0, k * z * z)
    return sigma * sigma


# ── SE(3) helpers ───────────────────────────────────────────────────────────
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ── registration ────────────────────────────────────────────────────────────
def rigid_align(src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None,
                with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch/Umeyama: best (R,t[,s]) mapping ``src`` onto ``dst`` (paired rows).

    Returns (R 3x3, t 3, s). Needs ≥3 non-degenerate correspondences.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(src))
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    wsum = max(w.sum(), 1e-9)
    mu_s = (w * src).sum(0) / wsum
    mu_d = (w * dst).sum(0) / wsum
    S = src - mu_s
    D = dst - mu_d
    H = (w * S).T @ D
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1.0, 1.0, d])
    R = Vt.T @ Dm @ U.T
    if with_scale:
        var = (w * (S ** 2)).sum() / wsum
        s = float((sig * np.array([1, 1, d])).sum() / max(var, 1e-9))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ransac_rigid(src: np.ndarray, dst: np.ndarray, thresh: float = 0.008, iters: int = 80,
                 min_inliers: int = 6, rng=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Robust rigid (R,t) from noisy 3D-3D correspondences via RANSAC + inlier refit.

    Used by the RGB-D feature odometry: aligned depth gives each RGB feature an exact 3D point,
    so frame↔canonical motion is a 3D-3D problem; RANSAC rejects bad LK matches / depth holes.
    Returns (R, t, inlier_ratio).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3:
        return np.eye(3), np.zeros(3), 0.0
    rng = rng or np.random.RandomState(0)
    best_inl = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t, _ = rigid_align(src[idx], dst[idx])
        resid = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inl = resid < thresh
        if inl.sum() > best_count:
            best_count, best_inl = int(inl.sum()), inl
    if best_count < min_inliers:
        R, t, _ = rigid_align(src, dst)        # fall back to plain LS on everything
        return R, t, best_count / max(n, 1)
    R, t, _ = rigid_align(src[best_inl], dst[best_inl])   # refit on inliers
    return R, t, best_count / float(n)


def point_to_plane_icp(src: np.ndarray, dst_pts: np.ndarray, dst_normals: np.ndarray,
                       T_init: Optional[np.ndarray] = None, max_iters: int = 20,
                       max_corr_dist: float = 0.05, tol: float = 1e-6,
                       dst_tree=None) -> Tuple[np.ndarray, float, float]:
    """Point-to-plane ICP aligning ``src`` (M,3) onto target cloud (``dst_pts``+normals).

    Returns (T 4x4 mapping src→dst, fitness=inlier_ratio, inlier_rmse[m]).
    Robust: Huber-style trimming via ``max_corr_dist``; nearest-neighbour via cKDTree.
    Pass a prebuilt ``dst_tree`` (cKDTree over ``dst_pts``) to avoid rebuilding it per frame.
    """
    src = np.asarray(src, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    dst_normals = np.asarray(dst_normals, dtype=np.float64)
    T = np.eye(4) if T_init is None else T_init.copy()
    if dst_tree is not None:
        tree = dst_tree
    elif cKDTree is not None and len(dst_pts):
        tree = cKDTree(dst_pts)
    else:
        tree = None
    fitness, rmse = 0.0, float("inf")
    if len(dst_pts) == 0 or len(src) == 0:
        return T, fitness, rmse

    for _ in range(max_iters):
        cur = transform_points(T, src)
        if tree is not None:
            dist, idx = tree.query(cur, k=1)
        else:  # brute force fallback
            d2 = ((cur[:, None, :] - dst_pts[None, :, :]) ** 2).sum(-1)
            idx = d2.argmin(1)
            dist = np.sqrt(d2[np.arange(len(cur)), idx])
        inl = dist < max_corr_dist
        if inl.sum() < 6:
            break
        p = cur[inl]
        q = dst_pts[idx[inl]]
        nrm = dst_normals[idx[inl]]
        # linearized point-to-plane: minimize Σ ((p+ [w]×p + tt - q)·n)^2 over (w,tt)
        A = np.concatenate([np.cross(p, nrm), nrm], axis=1)  # (m,6)
        b = -((p - q) * nrm).sum(1)                          # (m,)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        w, tt = x[:3], x[3:]
        theta = np.linalg.norm(w)
        if theta < 1e-12:
            dR = np.eye(3)
        else:
            k = w / theta
            K = _skew(k)
            dR = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        dT = make_T(dR, tt)
        T = dT @ T
        rmse = float(np.sqrt((((p - q) * nrm).sum(1) ** 2).mean()))
        fitness = float(inl.mean())
        if np.linalg.norm(x) < tol:
            break
    return T, fitness, rmse
