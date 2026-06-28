"""Canonical patient model — the long-term anchor (Approach B core).

Holds the auscultation points fixed in a **canonical patient frame** (anchored to anatomy at
init) plus a bounded surfel cloud of the back surface. Each frame it estimates the rigid
transform ``T_cam←canon`` that maps the canonical model into the live camera frame, separating
camera + rigid-patient motion from the (small, non-rigid) breathing that ``snap_to_surface``
absorbs. Memory is constant: the surfel cloud is voxel-downsampled to a fixed budget.

Pose estimation is two-stage and degrades gracefully:
  1. **Kabsch** on landmark correspondences (canonical pts ↔ current detected 3D pts) — strong,
     closed-form, available whenever a detection landed.
  2. **point-to-plane ICP** refine (canonical surfels → current depth cloud) — dense, uses the
     whole back surface; also the *only* estimator between detections (init from last pose).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .geometry import (invert_T, make_T, point_to_plane_icp, rigid_align,
                       transform_points, valid_points)

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # noqa: BLE001
    cKDTree = None  # type: ignore


def voxel_downsample(pts: np.ndarray, nrm: np.ndarray, voxel: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Average points (and normals) within each ``voxel``-sized cell."""
    if len(pts) == 0:
        return pts, nrm
    good = np.isfinite(pts).all(axis=1)        # drop non-finite rows: floor(nan).astype(int64) is a
    if not good.all():                          # garbage bucket that would leak a NaN surfel into the
        pts, nrm = pts[good], nrm[good]         # canonical surface and poison every downstream query
        if len(pts) == 0:
            return pts, nrm
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    M = inv.max() + 1
    sp = np.zeros((M, 3)); cnt = np.zeros(M)
    np.add.at(sp, inv, pts); np.add.at(cnt, inv, 1.0)
    sp /= cnt[:, None]
    sn = np.zeros((M, 3)); np.add.at(sn, inv, nrm)
    sn /= np.clip(np.linalg.norm(sn, axis=1, keepdims=True), 1e-9, None)
    return sp, sn


@dataclass
class PoseEstimate:
    T_cam_canon: np.ndarray            # 4x4 canonical → camera
    fitness: float                     # ICP inlier ratio (0..1)
    rmse: float                        # ICP inlier RMSE [m]
    method: str                        # "kabsch+icp" | "icp" | "kabsch" | "coast"


@dataclass
class CanonicalModel:
    voxel: float = 0.005               # 5 mm surfel resolution
    max_surfels: int = 20000           # hard memory cap
    fuse_rate: float = 0.1             # fraction of new surfels merged per fuse() call
    snap_radius: float = 0.02          # 2 cm neighbourhood for breathing snap

    points: Optional[np.ndarray] = None        # (10,3) canonical auscultation points
    addrs: List[Tuple[int, float, float]] = field(default_factory=list)  # IUV addr per point
    surfels: Optional[np.ndarray] = None       # (M,3) canonical surface
    normals: Optional[np.ndarray] = None       # (M,3)
    initialized: bool = False
    _tree: object = None                       # cached kdtree over surfels

    # -- initialization ---------------------------------------------------
    def initialize(self, points_canon: np.ndarray, surface_pts: np.ndarray,
                   surface_nrm: np.ndarray, addrs: List[Tuple[int, float, float]]) -> None:
        self.points = np.asarray(points_canon, dtype=np.float64).reshape(-1, 3)
        self.addrs = list(addrs)
        sp, sn = voxel_downsample(surface_pts, surface_nrm, self.voxel)
        self._set_surfels(sp, sn)
        self.initialized = True

    def _set_surfels(self, sp: np.ndarray, sn: np.ndarray) -> None:
        if len(sp) > self.max_surfels:
            keep = np.random.choice(len(sp), self.max_surfels, replace=False)
            sp, sn = sp[keep], sn[keep]
        self.surfels, self.normals = sp, sn
        self._tree = cKDTree(sp) if (cKDTree is not None and len(sp)) else None

    # -- pose estimation --------------------------------------------------
    def estimate_pose(self, cur_pts: np.ndarray, cur_nrm: np.ndarray,
                      T_init: Optional[np.ndarray],
                      corr_canon: Optional[np.ndarray] = None,
                      corr_cam: Optional[np.ndarray] = None,
                      icp_iters: int = 15, cur_tree=None) -> PoseEstimate:
        """Estimate ``T_cam←canon``. ``corr_*`` are matched landmark pairs (optional)."""
        T = np.eye(4) if T_init is None else T_init.copy()
        method = "coast"
        # 1) closed-form init from landmark correspondences
        if corr_canon is not None and len(corr_canon) >= 3:
            R, t, _ = rigid_align(corr_canon, corr_cam)
            T = make_T(R, t)
            method = "kabsch"
        # 2) dense ICP refine against the live cloud
        if self.surfels is not None and len(cur_pts) >= 50:
            T, fit, rmse = point_to_plane_icp(self.surfels, cur_pts, cur_nrm, T_init=T,
                                              max_iters=icp_iters, max_corr_dist=0.05,
                                              dst_tree=cur_tree)
            method = "kabsch+icp" if method == "kabsch" else "icp"
            return PoseEstimate(T, fit, rmse, method)
        if method == "kabsch":
            return PoseEstimate(T, 1.0, 0.0, "kabsch")
        return PoseEstimate(T, 0.0, float("inf"), "coast")

    # -- surface fusion (bounded memory) ----------------------------------
    def fuse(self, cur_pts_cam: np.ndarray, cur_nrm_cam: np.ndarray,
             T_cam_canon: np.ndarray) -> None:
        """Merge a fraction of the current depth cloud into the canonical surfel set."""
        if self.surfels is None or len(cur_pts_cam) == 0:
            return
        Tic = invert_T(T_cam_canon)                      # camera → canonical
        pc = transform_points(Tic, cur_pts_cam)
        nc = (Tic[:3, :3] @ cur_nrm_cam.T).T
        k = max(1, int(len(pc) * self.fuse_rate))
        sel = np.random.choice(len(pc), min(k, len(pc)), replace=False)
        merged = np.vstack([self.surfels, pc[sel]])
        mnrm = np.vstack([self.normals, nc[sel]])
        sp, sn = voxel_downsample(merged, mnrm, self.voxel)
        self._set_surfels(sp, sn)

    # -- breathing / non-rigid correction ---------------------------------
    def snap_to_surface(self, p_cam: np.ndarray, cur_tree, cur_pts: np.ndarray,
                        cur_nrm: np.ndarray) -> Tuple[np.ndarray, float]:
        """Project a predicted camera-frame point onto the live surface along its normal.

        Captures breathing (mostly normal-direction surface motion) without dragging the point
        laterally. Returns (corrected_point, |normal_displacement| [m]).
        """
        if cur_tree is None:
            return p_cam, 0.0
        idxs = cur_tree.query_ball_point(p_cam, self.snap_radius)
        if len(idxs) < 4:
            return p_cam, 0.0
        nbr = cur_pts[idxs]
        c = nbr.mean(0)
        n = cur_nrm[idxs].mean(0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            return p_cam, 0.0
        n = n / nn
        disp = float((p_cam - c) @ n)
        return p_cam - disp * n, abs(disp)
