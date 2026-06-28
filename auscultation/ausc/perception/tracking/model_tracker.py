"""Model-based deformable tracking for the 10 posterior-lung points.

Paradigm (vs per-frame re-detection): the 10 points are ONE near-rigid anatomical constellation,
not 10 independent detections. So each frame we estimate a single low-DOF body transform
(similarity: rotation + uniform scale + translation) fit to ALL 10 detector points at once
(weighted Umeyama with IRLS outlier rejection). Ten noisy observations collapse into ~7 pose
parameters -> per-point noise is averaged down and the relative jitter vanishes by construction.
The pose is then temporally low-passed (the jitter is >2 Hz, real motion <1 Hz, so this removes
flicker without lag). Output = canonical template warped by the smoothed pose + a small, heavily
regularized per-point residual that absorbs slow non-rigid deformation (breathing / posture).

This is "3D-based tracking" in the model sense: we track a known 3D deformable template using the
detector as a *noise-averaged pose sensor* — not dense SLAM (the back is textureless + deformable)
and not 10 independent point filters (which pass the noise straight through).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

from .detector import LUNG_ORDER
from .geometry import (Intrinsics, body_band_mask, normals_from_pointmap, pointmap_from_depth,
                       sample_depth_m, valid_points)

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None

STATUS_INIT, STATUS_TRACKING, STATUS_COASTING, STATUS_LOST = "INIT", "TRACKING", "COASTING", "LOST"


@dataclass
class ModelConfig:
    init_frames: int = 12          # confident frames to build the canonical template
    min_conf: float = 0.3          # min detector frame-confidence to use a detection
    min_valid_pts: int = 6         # min points with valid depth to fit a pose
    pose_ema: float = 0.4          # temporal low-pass on the pose (lower = smoother, more lag)
    resid_ema: float = 0.04        # very slow per-point non-rigid residual
    max_resid_m: float = 0.04      # clamp the residual (4 cm)
    huber_m: float = 0.02          # IRLS inlier scale (2 cm) for robust pose fit
    irls_iters: int = 3
    coast_max: int = 20            # predict-only frames before a point goes LOST
    max_depth_m: float = 3.0
    base_std_m: float = 0.003      # position uncertainty of a freshly-measured point [m]
    coast_growth_m: float = 0.003  # uncertainty added per frame a point is extrapolated (not measured)
    inlier_huber_mult: float = 1.5 # a fit point is "measured" if its residual < inlier_huber_mult·huber_m
    # ── depth surface-snap (lift z via local back-plane, not single-pixel depth) ──
    # Default OFF: on real clips it only improves depth noise 0.3→0.2 mm but adds a per-frame
    # pointcloud+KD-tree build (~costly at 1080p). Enable for offline/quality runs.
    surface_snap: bool = False     # snap each ray onto the local back-surface plane
    icp_stride: int = 2            # depth downsample for the body cloud / KD-tree
    body_band_m: float = 0.25      # depth band around the patient for cloud masking
    snap_radius_m: float = 0.03    # neighbourhood radius for the local-plane fit
    snap_sanity_m: float = 0.03    # reject a snap whose depth disagrees with the prediction by >this


@dataclass
class MTPoint:
    name: str
    xyz_camera: np.ndarray
    pixel: np.ndarray
    status: str
    measured: bool = False              # was this point actually OBSERVED this frame (detector inlier),
    #                                    not silently extrapolated from the rest of the constellation?
    position_std: float = 0.0           # position uncertainty [m] (grows while extrapolated/coasting)
    xyz_world: Optional[np.ndarray] = None  # camera→base extrinsics applied (None if no extrinsics)


@dataclass
class MTOutput:
    t: float
    initialized: bool
    points: List[MTPoint]
    pose_fit_mm: float = 0.0
    n_tracking: int = 0
    n_snapped: int = 0             # of 10 points: how many got a surface-plane snap this frame
    n_measured: int = 0           # of 10 points: how many were actually observed (not extrapolated)


def _umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    """Weighted similarity (R, s, t) with dst ≈ s·R·src + t (Umeyama 1991)."""
    W = w / max(w.sum(), 1e-9)
    mu_s = (src * W[:, None]).sum(0)
    mu_d = (dst * W[:, None]).sum(0)
    s0 = src - mu_s
    d0 = dst - mu_d
    Sigma = (d0 * W[:, None]).T @ s0
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (W * (s0 ** 2).sum(1)).sum()
    s = float((D * np.diag(S)).sum() / max(var_s, 1e-9))
    t = mu_d - s * R @ mu_s
    return R, s, t


class ModelBasedTracker:
    """Drop-in tracker with the same ``process(rgb, depth, t, depth_scale)`` contract."""

    def __init__(self, detector, intrinsics: Intrinsics, config: Optional[ModelConfig] = None,
                 extrinsics: Optional[np.ndarray] = None):
        self.detector = detector
        self.intr = intrinsics
        self.cfg = config or ModelConfig()
        self.extrinsics = extrinsics                        # 4x4 camera→base (robot frame), optional
        self.canonical: Optional[np.ndarray] = None        # (10,3) body-frame template
        self.residual = np.zeros((len(LUNG_ORDER), 3))      # (10,3) body-frame slow deformation
        self.R = np.eye(3); self.s = 1.0; self.t = np.zeros(3)
        self._have_pose = False
        self._init_obs: List[np.ndarray] = []
        self._misses = 0
        self._miss_pp = np.zeros(len(LUNG_ORDER), dtype=int)  # per-point frames-since-measured
        self._initialized = False
        self._cur_pts = None                                # body-band cloud (for surface-snap)
        self._cur_tree = None
        self._n_snapped = 0

    # -- body cloud + ray surface-snap (Step 1: stable depth at the source) ------
    def _build_cloud(self, depth):
        if not self.cfg.surface_snap or cKDTree is None:
            self._cur_pts = self._cur_tree = None
            return
        P = pointmap_from_depth(depth, self.intr, stride=self.cfg.icp_stride, max_m=self.cfg.max_depth_m)
        P[~body_band_mask(P, band=self.cfg.body_band_m)] = np.nan
        pts, _ = valid_points(P, normals_from_pointmap(P))
        self._cur_pts = pts
        self._cur_tree = cKDTree(pts) if len(pts) else None

    def _snap_rays(self, uv, depth, seed_pred):
        """Lift pixels to 3D by intersecting each camera ray with the LOCAL back-surface plane
        (neighbourhood-averaged), not the single-pixel depth. Only z along the fixed ray changes,
        so the lateral (x,y) detection is preserved. Seeds the neighbourhood from the per-pixel
        depth, falling back to the pose prediction on holes; rejects a snap that disagrees with
        the prediction by > snap_sanity_m (curvature/grazing guard)."""
        naive = self._unproject(uv, depth)
        if self._cur_tree is None:
            return naive, 0
        out = naive.copy()
        rays = self.intr.unproject(uv, np.ones(len(uv)))     # ray dirs (z=1)
        nsnap = 0
        for i in range(len(uv)):
            seed = naive[i]
            if not np.isfinite(seed).all() and seed_pred is not None:
                seed = seed_pred[i]                          # hole → seed from the track
            if not np.isfinite(seed).all():
                continue
            idx = self._cur_tree.query_ball_point(seed, r=self.cfg.snap_radius_m)
            if len(idx) < 8:
                dd, ii = self._cur_tree.query(seed, k=16)
                idx = np.atleast_1d(ii)[np.atleast_1d(dd) < 2 * self.cfg.snap_radius_m]
            if len(idx) < 6:
                continue
            nbr = self._cur_pts[idx]; c = nbr.mean(0)
            _, _, vt = np.linalg.svd(nbr - c)
            n = vt[2]; ray = rays[i]; den = float(ray @ n)
            if abs(den) < 1e-6:
                continue
            s = float((c @ n) / den)
            if not (0.1 < s < self.cfg.max_depth_m):
                continue
            if seed_pred is not None and np.isfinite(seed_pred[i]).all() \
                    and abs(s - seed_pred[i][2]) > self.cfg.snap_sanity_m:
                continue                                     # snapped to a wrong plane → keep prediction/naive
            out[i] = ray * s; nsnap += 1
        return out, nsnap

    def _unproject(self, uv, depth):
        z = sample_depth_m(depth, uv, self.intr.depth_scale, max_m=self.cfg.max_depth_m)
        z = np.where(np.isfinite(z), z, np.nan)
        return self.intr.unproject(uv, z)

    # -- detection → 3D ---------------------------------------------------
    def _detect3d(self, rgb, depth):
        det = self.detector.detect(rgb, depth, self.intr)
        if det is None or det.confidence < self.cfg.min_conf:
            return None
        full = det.crop.det_to_full(det.xy_det)
        if not self.cfg.surface_snap or self._cur_tree is None:
            self._n_snapped = 0
            return self._unproject(full, depth)
        seed_pred = self._pose_apply(self.canonical + self.residual) if self._have_pose else None
        obs, self._n_snapped = self._snap_rays(full, depth, seed_pred)
        return obs                                           # (10,3), NaN where no depth/snap

    # -- canonical build (generalized Procrustes over init frames) --------
    def _build_canonical(self):
        ref = self._init_obs[0]
        acc = np.zeros_like(ref); cnt = np.zeros(len(ref))
        for obs in self._init_obs:
            m = np.isfinite(ref).all(1) & np.isfinite(obs).all(1)
            if m.sum() < self.cfg.min_valid_pts:
                continue
            R, s, t = _umeyama(obs[m], ref[m], np.ones(m.sum()))
            al = (s * (R @ obs.T).T + t)
            ok = np.isfinite(al).all(1)
            acc[ok] += al[ok]; cnt[ok] += 1
        canon = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), ref)
        self.canonical = canon
        self.R, self.s, self.t = np.eye(3), 1.0, np.zeros(3)
        self._have_pose = True

    # -- robust pose fit (IRLS) -------------------------------------------
    def _fit_pose(self, obs):
        m = np.isfinite(obs).all(1) & np.isfinite(self.canonical).all(1)
        if m.sum() < self.cfg.min_valid_pts:
            return None
        src = self.canonical[m]; dst = obs[m]
        w = np.ones(m.sum())
        R = s = t = None
        for _ in range(self.cfg.irls_iters):
            R, s, t = _umeyama(src, dst, w)
            r = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
            w = 1.0 / np.maximum(r / self.cfg.huber_m, 1.0)     # Huber-ish
        # hard-inlier refit: drop residual outliers entirely so they can't pull the pose (a few
        # transient detector glitches must not corrupt the other points)
        inl = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1) < self.cfg.inlier_huber_mult * self.cfg.huber_m
        if int(inl.sum()) >= self.cfg.min_valid_pts:
            R, s, t = _umeyama(src[inl], dst[inl], np.ones(int(inl.sum())))
        resid = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
        fit = float(np.median(resid))
        # a point is MEASURED iff it had a valid observation AND is a fit inlier (not an outlier
        # the IRLS down-weighted). Points outside `m` were never observed → extrapolated → not measured.
        measured = np.zeros(len(LUNG_ORDER), dtype=bool)
        measured[np.flatnonzero(m)] = resid < self.cfg.inlier_huber_mult * self.cfg.huber_m
        return R, s, t, measured, fit

    def _smooth_pose(self, R, s, t):
        a = self.cfg.pose_ema
        # rotation: low-pass via the relative rotvec
        if cv2 is not None:
            rel = self.R.T @ R
            rv, _ = cv2.Rodrigues(rel)
            Rstep, _ = cv2.Rodrigues(a * rv)
            self.R = self.R @ Rstep
        else:
            self.R = R
        self.s = (1 - a) * self.s + a * s
        self.t = (1 - a) * self.t + a * t

    def _pose_apply(self, body_pts):
        return (self.s * (self.R @ body_pts.T).T + self.t)

    def _pose_inv(self, cam_pts):
        return (self.R.T @ (cam_pts - self.t).T).T / self.s

    # -- main entry -------------------------------------------------------
    def process(self, rgb, depth, t: float = 0.0, depth_scale: Optional[float] = None) -> MTOutput:
        if depth_scale is not None and abs(depth_scale - self.intr.depth_scale) > 1e-9:
            import dataclasses
            self.intr = dataclasses.replace(self.intr, depth_scale=depth_scale)
        self._n_snapped = 0
        if depth is not None:
            self._build_cloud(depth)                         # body cloud for ray surface-snap
        obs = None if (rgb is None or depth is None) else self._detect3d(rgb, depth)

        if not self._initialized:
            if obs is not None and np.isfinite(obs).all(1).sum() >= self.cfg.min_valid_pts:
                self._init_obs.append(obs)
            if len(self._init_obs) >= self.cfg.init_frames:
                self._build_canonical()
                self._initialized = True
            return self._emit(t, init_only=True)

        fitres = self._fit_pose(obs) if obs is not None else None
        if fitres is None:
            self._misses += 1
            self._miss_pp += 1                               # no detection → every point extrapolated
            return self._emit(t, fit_mm=0.0)                 # hold last smoothed pose, per-point status
        self._misses = 0
        R, s, tt, measured, fit = fitres
        self._miss_pp = np.where(measured, 0, self._miss_pp + 1)
        self._smooth_pose(R, s, tt)
        # slow per-point residual in body frame (absorbs non-rigid deformation, clamped). Only update
        # from MEASURED points — never let an extrapolated point feed its own (circular) residual.
        resid_obs_body = self._pose_inv(obs) - self.canonical
        upd = measured & np.isfinite(resid_obs_body).all(1)
        self.residual[upd] = (1 - self.cfg.resid_ema) * self.residual[upd] + self.cfg.resid_ema * resid_obs_body[upd]
        n = np.linalg.norm(self.residual, axis=1, keepdims=True)
        self.residual = np.where(n > self.cfg.max_resid_m, self.residual / np.maximum(n, 1e-9) * self.cfg.max_resid_m, self.residual)
        return self._emit(t, fit_mm=fit * 1000.0)

    def _emit(self, t, fit_mm: float = 0.0, init_only: bool = False) -> MTOutput:
        if not self._initialized:
            pts = [MTPoint(nm, np.full(3, np.nan), np.full(2, np.nan), STATUS_INIT) for nm in LUNG_ORDER]
            return MTOutput(t, False, pts)
        cam = self._pose_apply(self.canonical + self.residual)
        pts = []
        ntr = nmeas = 0
        for k, nm in enumerate(LUNG_ORDER):
            c = cam[k]
            pix = self.intr.project(c[None])[0] if c[2] > 1e-3 else np.full(2, np.nan)
            miss = int(self._miss_pp[k])
            measured = miss == 0                                  # observed this very frame
            if measured:
                st = STATUS_TRACKING
            elif miss <= self.cfg.coast_max:
                st = STATUS_COASTING                             # extrapolated from the constellation
            else:
                st = STATUS_LOST                                 # too long unobserved → don't trust
            std = self.cfg.base_std_m + miss * self.cfg.coast_growth_m
            world = None
            if self.extrinsics is not None and np.isfinite(c).all():
                world = (self.extrinsics[:3, :3] @ c + self.extrinsics[:3, 3])
            ntr += int(st == STATUS_TRACKING); nmeas += int(measured)
            pts.append(MTPoint(nm, c, pix, st, measured=measured, position_std=std, xyz_world=world))
        return MTOutput(t, True, pts, pose_fit_mm=fit_mm, n_tracking=ntr,
                        n_snapped=self._n_snapped, n_measured=nmeas)
