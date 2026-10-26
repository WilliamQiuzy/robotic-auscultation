"""StreamingAuscultationTracker — the production streaming pipeline (Approach B).

Per frame it fuses four independent evidence channels into a stable, drift-free, anatomically
anchored 3D estimate of the 10 posterior-lung auscultation points:

  D detector  (throttled)  absolute 2D pts → depth → 3D  ⟶ EKF measurement (kills drift)
  F flow      (per frame)  LK 2D track → depth → 3D       ⟶ EKF measurement (smooth, high-rate)
  S semantic  (with detect) IUV (part,U,V) addr → 3D      ⟶ EKF measurement (pose-invariant)
  G geometry  (per frame)  point-to-plane ICP ⟶ T_cam←canon : converts D/F/S into the canonical
                            frame and places the EKF estimate back into the live camera frame.

The EKF lives in the **canonical patient frame** (anatomically fixed). Output camera points are
``T_cam←canon · ekf.position`` then snapped to the live surface for the current breathing phase.
Memory is constant in session length (see CanonicalModel + one-frame LK state).

Design: ``docs/perception_streaming_tracker.md``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .canonical import CanonicalModel, PoseEstimate
from .detector import LandmarkDetector, LUNG_ORDER, iuv_locate
from .ekf import EKFConfig, PointEKF
from .flow import FlowResult, LKFlow
from .geometry import (Intrinsics, axial_noise_var, body_band_mask, invert_T,
                       normals_from_pointmap, pointmap_from_depth, sample_depth_m,
                       transform_points, valid_points)
from .rgbd_odometry import FeatureField

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # noqa: BLE001
    cKDTree = None  # type: ignore

STATUS_INIT, STATUS_TRACKING, STATUS_COASTING, STATUS_LOST = "INIT", "TRACKING", "COASTING", "LOST"


@dataclass(frozen=True)
class TrackedPoint:
    name: str
    xyz_canonical: np.ndarray          # (3,) anatomical anchor in canonical frame [m]
    xyz_camera: np.ndarray             # (3,) live skin point in camera frame [m]
    pixel: np.ndarray                  # (2,) full-frame pixel of xyz_camera
    confidence: float                  # 0..1 (1/(1+pos_std), scaled)
    status: str
    velocity: np.ndarray               # (3,) canonical-frame velocity [m/s]
    position_std: float                # EKF position uncertainty [m]
    xyz_world: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TrackerOutput:
    t: float
    frame_index: int
    initialized: bool
    points: List[TrackedPoint]
    T_cam_canon: Optional[np.ndarray]
    icp_fitness: float
    icp_rmse: float
    detector_ran: bool
    detector_age_s: float
    n_tracking: int
    n_lost: int
    vo_fitness: float = 0.0                 # RGB-D feature-odometry inlier ratio
    respiratory_offset_m: float = 0.0       # signed breathing displacement of the back [m]
    respiratory_amplitude_m: float = 0.0    # running breathing amplitude [m]


@dataclass
class TrackerConfig:
    detect_interval_s: float = 0.5     # throttle the heavy detector (≈2 Hz)
    icp_stride: int = 6                # depth downsample for ICP/snap clouds
    fuse_interval: int = 10            # fuse surface into canonical model every N frames
    use_feature_odometry: bool = True  # RGB-D feature VO for tangential lock (smooth back)
    body_band_m: float = 0.25          # depth band around patient for body masking [m]
    depth_aware_var: bool = True       # scale EKF measurement noise by axial depth noise
    resp_ema: float = 0.02             # EMA rate for the respiratory baseline (slow)
    surface_gate_mm: float = 40.0      # max plane-residual at init (a back is gently curved)
    surface_snap_meas: bool = True     # snap detector/flow rays onto the local back surface
    snap_radius_m: float = 0.03        # neighbourhood radius for the local-plane fit [m]
    # ── track-primary policy ──────────────────────────────────────────────────
    # The detector (perc_lmh_v4) is the *least* reliable component, so we lean on the
    # training-free geometric/photometric tracking and use the detector only defensively:
    # confirm the anchor across frames, re-detect on loss, and accept a correction only if it
    # agrees with the track. A single bad-pose detection frame can no longer move the estimate.
    init_confirm_frames: int = 3       # consecutive consistent detections required to anchor
    init_consistency_mm: float = 14.0  # max structure drift between consecutive init detections
    drift_audit_s: float = 1.0         # steady-state detector cadence (safe: every detection is
                                       # agreement-gated, so more frequent only helps drift)
    detect_conf_accept: float = 0.5    # min detector confidence to accept its correction
    detection_agree_mm: float = 18.0   # detection must agree with the track within this to be used
    var_det: float = 2.5e-5            # (5 mm)^2 detector measurement variance [m^2]
    var_flow: float = 6.4e-5           # (8 mm)^2 flow measurement variance
    var_iuv: float = 1.6e-4            # (12.6 mm)^2 IUV-anchor measurement variance
    min_conf_corr: float = 0.3         # min detector confidence to use for Kabsch / re-anchor
    max_misses_coast: int = 20         # predict-only steps before COASTING→LOST consideration
    lost_std_m: float = 0.05           # position uncertainty that marks a point LOST
    force_redetect_lost: int = 3       # ≥this many LOST points forces an early detection
    max_depth_m: float = 3.0
    ekf: EKFConfig = field(default_factory=EKFConfig)


class StreamingAuscultationTracker:
    """Long-duration 3D tracker for the 10 posterior-lung auscultation points."""

    def __init__(self, detector: Optional[LandmarkDetector], intrinsics: Intrinsics,
                 config: Optional[TrackerConfig] = None,
                 extrinsics: Optional[np.ndarray] = None,
                 addrs: Optional[List[Tuple[int, float, float]]] = None) -> None:
        self.detector = detector
        self.intr = intrinsics
        self.cfg = config or TrackerConfig()
        self.extrinsics = extrinsics          # 4x4 camera→world (optional)
        self.canonical = CanonicalModel()
        self.ekfs: List[Optional[PointEKF]] = [None] * len(LUNG_ORDER)
        self.statuses: List[str] = [STATUS_LOST] * len(LUNG_ORDER)
        self._lk = LKFlow()
        self._prev_gray: Optional[np.ndarray] = None
        self._pixels = np.full((len(LUNG_ORDER), 2), np.nan)   # full-frame tracked pixels
        self._T = np.eye(4)                                     # last T_cam←canon
        self._last_detect_t = -1e9
        self._last_t: Optional[float] = None
        self._frame = 0
        self._addrs = addrs                                    # IUV addrs (fallback to detector)
        self._field = (FeatureField(intrinsics, band=self.cfg.body_band_m)
                       if self.cfg.use_feature_odometry else None)
        self._vo_fit = 0.0
        self._resp_base: Optional[float] = None               # respiratory baseline depth [m]
        self._resp_amp = 0.0
        self._resp_off = 0.0
        self._init_count = 0                                   # confirmed-init streak
        self._init_prev3d: Optional[np.ndarray] = None
        self._cur_pts: Optional[np.ndarray] = None             # latest body-band cloud (for snap)
        self._cur_tree = None

    # -- helpers ----------------------------------------------------------
    def _set_depth_scale(self, depth_scale: Optional[float]) -> None:
        if depth_scale is not None and abs(depth_scale - self.intr.depth_scale) > 1e-9:
            self.intr = dataclasses.replace(self.intr, depth_scale=depth_scale)
            if self._field is not None:
                self._field.intr = self.intr

    def _unproject_pixels(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """(N,2) full-frame pixels + depth map → (N,3) camera points [m]; NaN where no depth."""
        z = sample_depth_m(depth, uv, self.intr.depth_scale, max_m=self.cfg.max_depth_m)
        out = np.full((len(uv), 3), np.nan)
        ok = np.isfinite(z) & np.isfinite(uv).all(axis=1)
        if ok.any():
            out[ok] = self.intr.unproject(uv[ok], z[ok])
        return out

    def _build_cloud(self, depth: np.ndarray):
        P = pointmap_from_depth(depth, self.intr, stride=self.cfg.icp_stride,
                                max_m=self.cfg.max_depth_m)
        P[~body_band_mask(P, band=self.cfg.body_band_m)] = np.nan   # keep patient, drop bed/bg
        N = normals_from_pointmap(P)
        pts, nrm = valid_points(P, N)
        tree = cKDTree(pts) if (cKDTree is not None and len(pts)) else None
        return pts, nrm, tree

    # -- main entry -------------------------------------------------------
    def process(self, rgb_full: Optional[np.ndarray], depth: Optional[np.ndarray],
                t: Optional[float] = None, depth_scale: Optional[float] = None) -> TrackerOutput:
        if t is None:
            t = time.monotonic()
        elif not np.isfinite(t):
            # storing it would make every later dt non-finite and stop the drift audit for good
            raise ValueError(f"frame timestamp must be finite, got {t!r}")
        self._set_depth_scale(depth_scale)
        self._frame += 1
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 1e-3)
        self._last_t = t
        gray = self._lk.to_gray(rgb_full) if rgb_full is not None else None

        if depth is None:
            return self._emit(t, detector_ran=False, fit=0.0, rmse=float("inf"))

        cur_pts, cur_nrm, cur_tree = self._build_cloud(depth)
        self._cur_pts, self._cur_tree = cur_pts, cur_tree     # expose for ray surface-snap

        # run the (heavy, less-reliable) detector sparingly — track-primary policy:
        # every frame only until the anchor is confirmed; then only on loss or a slow audit.
        n_lost = sum(s == STATUS_LOST for s in self.statuses)
        if not self.canonical.initialized:
            want_detect = True
        else:
            want_detect = (n_lost >= self.cfg.force_redetect_lost
                           or (t - self._last_detect_t) >= self.cfg.drift_audit_s)
        det = None
        det3d = iuv3d = None
        if want_detect and self.detector is not None and rgb_full is not None:
            det = self.detector.detect(rgb_full, depth, self.intr)
            if det is not None:
                self._last_detect_t = t
                det3d, iuv3d = self._detection_3d(det, depth)

        if not self.canonical.initialized:
            self._update_respiratory(cur_pts)
            confirmed = self._confirm_init(det, det3d)        # require N consistent detections
            ok = (self._try_initialize(det, det3d, gray, depth, cur_pts, cur_nrm)
                  if confirmed else False)
            return self._emit(t, detector_ran=det is not None, fit=1.0 if ok else 0.0,
                              rmse=0.0, cur_pts=cur_pts, cur_nrm=cur_nrm, cur_tree=cur_tree)

        # whole-detection agreement gate: trust the track, not the detector. Use a detection only
        # if it is confident AND agrees with the current track — or the track is already LOST and
        # we need the detector to re-acquire identity. A bad-pose detection is otherwise ignored.
        accept_det = False
        if det3d is not None and det is not None:
            if n_lost >= self.cfg.force_redetect_lost:
                accept_det = True
            elif det.confidence >= self.cfg.detect_conf_accept and self._detection_agrees(det3d):
                accept_det = True
        if not accept_det:
            det3d = iuv3d = None                              # ignore an untrusted detection

        # --- per-frame update -------------------------------------------
        self._update_respiratory(cur_pts)
        flow = self._run_flow(gray)
        # G — RGB-D feature odometry: tangentially-locked T init (fixes smooth-back ICP slide)
        T_vo = self._T
        if self._field is not None and gray is not None:
            T_vo, self._vo_fit, _ = self._field.estimate(gray, depth, self._T)
        pose = self._estimate_pose(det, det3d, cur_pts, cur_nrm, cur_tree, T_init=T_vo)
        self._T = pose.T_cam_canon
        Tic = invert_T(self._T)

        accept_det = det3d is not None      # gate already nulled det3d if untrusted
        for k in range(len(LUNG_ORDER)):
            ekf = self.ekfs[k]
            if ekf is None:
                continue
            # re-acquire a LOST point from a trusted detection (the one time we lean on it)
            reacq = (accept_det and self.statuses[k] == STATUS_LOST
                     and np.isfinite(det3d[k]).all())
            if reacq:
                ekf = self.ekfs[k] = PointEKF(transform_points(Tic, det3d[k])[0], self.cfg.ekf)
            else:
                ekf.predict(dt)
            accepted = reacq
            # F — flow measurement
            if flow is not None and flow.valid[k]:
                obs = self._unproject_pixels(flow.pts[k:k + 1], depth)[0]
                if np.isfinite(obs).all():
                    accepted |= ekf.update(transform_points(Tic, obs)[0],
                                           self._meas_var(self.cfg.var_flow, obs[2]))
            # D — detector measurement
            if det3d is not None and np.isfinite(det3d[k]).all() and det.confidence >= 0.05:
                var = self._meas_var(self.cfg.var_det / max(det.confidence, 0.05), det3d[k][2])
                accepted |= ekf.update(transform_points(Tic, det3d[k])[0], var)
            # S — IUV semantic anchor
            if iuv3d is not None and np.isfinite(iuv3d[k]).all():
                accepted |= ekf.update(transform_points(Tic, iuv3d[k])[0],
                                       self._meas_var(self.cfg.var_iuv, iuv3d[k][2]))
            if not accepted:
                ekf.mark_miss()
            self._update_status(k)

        # refresh the pixels LK tracks: re-ground on the detection only if it was trusted
        self._refresh_pixels(det if accept_det else None, flow)
        self._prev_gray = gray

        # slow surface fusion keeps the canonical model current (throttled, bounded memory)
        if pose.fitness > 0.3 and self._frame % self.cfg.fuse_interval == 0:
            self.canonical.fuse(cur_pts, cur_nrm, self._T)

        return self._emit(t, detector_ran=det is not None, fit=pose.fitness, rmse=pose.rmse,
                          cur_pts=cur_pts, cur_nrm=cur_nrm, cur_tree=cur_tree)

    def _snap_rays(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift pixels to 3D by intersecting each camera ray with the LOCAL back-surface plane,
        not by reading the single-pixel depth. Robust to holes/edges and removes the along-ray
        depth error that silently corrupts the EKF's lateral (x,y) estimate. Seeds the plane
        neighbourhood from the per-pixel depth, falling back to the current EKF prediction so a
        hole / off-back pixel still snaps to where the point is tracked on the back."""
        naive = self._unproject_pixels(uv, depth)
        if not self.cfg.surface_snap_meas or self._cur_tree is None or self._cur_pts is None:
            return naive
        out = naive.copy()
        rays = self.intr.unproject(uv, np.ones(len(uv)))      # ray dirs (z=1)
        for i in range(len(uv)):
            seed = naive[i]
            if not np.isfinite(seed).all():                   # hole → seed from the track
                ek = self.ekfs[i]
                if ek is not None and self.canonical.initialized:
                    seed = transform_points(self._T, ek.position[None])[0]
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
            if 0.1 < s < self.cfg.max_depth_m:
                out[i] = ray * s
        return out

    # -- detection → 3D ---------------------------------------------------
    def _detection_3d(self, det, depth) -> Tuple[np.ndarray, np.ndarray]:
        full = det.crop.det_to_full(det.xy_det)
        det3d = self._snap_rays(full, depth)
        iuv3d = np.full((len(LUNG_ORDER), 3), np.nan)
        addrs = self._resolve_addrs()
        if det.part is not None and det.uv is not None and addrs is not None:
            for k, addr in enumerate(addrs):
                loc = iuv_locate(det.part, det.uv, addr)
                if loc is not None:
                    fp = det.crop.det_to_full(np.array([loc]))
                    iuv3d[k] = self._unproject_pixels(fp, depth)[0]
        return det3d, iuv3d

    def _resolve_addrs(self) -> Optional[List[Tuple[int, float, float]]]:
        if self._addrs is not None:
            return self._addrs
        try:  # the authoritative posterior-lung IUV addresses
            from humman_label import LUNG_ADDR_V6A  # type: ignore
            return [LUNG_ADDR_V6A[nm] for nm in LUNG_ORDER]
        except Exception:  # noqa: BLE001
            return None

    # -- initialization ---------------------------------------------------
    def _confirm_init(self, det, det3d) -> bool:
        """Require ``init_confirm_frames`` consecutive, structurally-consistent, confident
        detections before anchoring the canonical model — so one bad frame-1 can't poison it."""
        if det is None or det3d is None:
            self._init_count = 0; self._init_prev3d = None
            return False
        valid = np.isfinite(det3d).all(axis=1)
        if (valid.sum() < 6 or det.confidence < self.cfg.min_conf_corr
                or not self._on_surface(det3d, self.cfg.surface_gate_mm)):
            self._init_count = 0; self._init_prev3d = None
            return False
        if self._init_prev3d is not None:
            m = valid & np.isfinite(self._init_prev3d).all(axis=1)
            if m.sum() >= 3:
                from .geometry import rigid_align
                R, tt, _ = rigid_align(self._init_prev3d[m], det3d[m])
                resid = np.median(np.linalg.norm((R @ self._init_prev3d[m].T).T + tt - det3d[m], axis=1))
                if resid > self.cfg.init_consistency_mm / 1000.0:
                    self._init_count = 1; self._init_prev3d = det3d   # restart streak here
                    return False
        self._init_count += 1
        self._init_prev3d = det3d
        return self._init_count >= self.cfg.init_confirm_frames

    def _detection_agrees(self, det3d) -> bool:
        """Does the detection agree with the current track? Compares detected 3D to the tracked
        camera-frame positions of currently-tracked points. Conservative: if too few points are
        comparable, return False (do not trust the detector by default)."""
        d = []
        for k in range(len(LUNG_ORDER)):
            ekf = self.ekfs[k]
            if ekf is None or self.statuses[k] == STATUS_LOST or not np.isfinite(det3d[k]).all():
                continue
            cam = transform_points(self._T, ekf.position[None])[0]
            d.append(float(np.linalg.norm(cam - det3d[k])))
        if len(d) < 4:
            return False
        return float(np.median(d)) < self.cfg.detection_agree_mm / 1000.0

    def _try_initialize(self, det, det3d, gray, depth, cur_pts, cur_nrm) -> bool:
        if det is None or det3d is None:
            return False
        valid = np.isfinite(det3d).all(axis=1)
        if valid.sum() < 6 or det.confidence < self.cfg.min_conf_corr:
            return False
        if not self._on_surface(det3d, self.cfg.surface_gate_mm):   # depth-consistency gate
            return False
        # canonical frame := camera frame at init; fill any missing point from its neighbours
        pts = det3d.copy()
        if not valid.all():
            mean = np.nanmean(det3d[valid], axis=0)
            pts[~valid] = mean
        addrs = self._resolve_addrs() or [(1, 0.5, 0.5)] * len(LUNG_ORDER)
        self.canonical.initialize(pts, cur_pts, cur_nrm, addrs)
        for k in range(len(LUNG_ORDER)):
            self.ekfs[k] = PointEKF(pts[k], self.cfg.ekf)
            self.statuses[k] = STATUS_INIT
        self._T = np.eye(4)
        self._pixels = det.crop.det_to_full(det.xy_det)
        self._prev_gray = gray
        if self._field is not None and gray is not None:           # seed the VO feature field
            self._field.reset(gray, depth=depth, T_cam_canon=np.eye(4))
        return True

    # -- per-frame sub-steps ---------------------------------------------
    def _run_flow(self, gray) -> Optional[FlowResult]:
        if gray is None or self._prev_gray is None:
            return None
        return self._lk.track(self._prev_gray, gray, self._pixels)

    def _estimate_pose(self, det, det3d, cur_pts, cur_nrm, cur_tree, T_init=None) -> PoseEstimate:
        corr_c = corr_x = None
        if (det is not None and det3d is not None and det.confidence >= self.cfg.min_conf_corr
                and self.canonical.points is not None):
            valid = np.isfinite(det3d).all(axis=1)
            if valid.sum() >= 3:
                corr_c = self.canonical.points[valid]
                corr_x = det3d[valid]
        return self.canonical.estimate_pose(
            cur_pts, cur_nrm, T_init=self._T if T_init is None else T_init,
            corr_canon=corr_c, corr_cam=corr_x, cur_tree=cur_tree)

    def _meas_var(self, base: float, obs_z: float) -> float:
        """Measurement variance floored by the depth-camera axial noise at this range."""
        if self.cfg.depth_aware_var and np.isfinite(obs_z):
            return max(base, float(axial_noise_var(obs_z)))
        return base

    def _update_respiratory(self, cur_pts) -> None:
        """Track the back's mean depth → breathing baseline / offset / amplitude."""
        if cur_pts is None or len(cur_pts) < 16:
            return
        z = float(np.median(cur_pts[:, 2]))
        if self._resp_base is None:
            self._resp_base = z
        self._resp_base = (1 - self.cfg.resp_ema) * self._resp_base + self.cfg.resp_ema * z
        off = z - self._resp_base
        self._resp_amp = 0.9 * self._resp_amp + 0.1 * abs(off)
        self._resp_off = off

    @staticmethod
    def _on_surface(pts3d: np.ndarray, tol_mm: float) -> bool:
        """Depth-consistency gate: do the valid detected points lie on a smooth plane patch?"""
        v = pts3d[np.isfinite(pts3d).all(axis=1)]
        if len(v) < 6:
            return False
        c = v.mean(0)
        _, _, vt = np.linalg.svd(v - c)
        resid = np.abs((v - c) @ vt[2])              # distance to best-fit plane
        return float(np.median(resid)) < tol_mm / 1000.0

    def _refresh_pixels(self, det, flow) -> None:
        new = self._pixels.copy()
        if flow is not None:
            new[flow.valid] = flow.pts[flow.valid]
        # project current EKF estimates for points whose flow was lost (lets LK re-acquire)
        for k in range(len(LUNG_ORDER)):
            if (flow is None or not flow.valid[k]) and self.ekfs[k] is not None:
                cam = transform_points(self._T, self.ekfs[k].position[None])[0]
                if cam[2] > 1e-3:
                    new[k] = self.intr.project(cam[None])[0]
        if det is not None and det.confidence >= self.cfg.min_conf_corr:
            new = det.crop.det_to_full(det.xy_det)   # re-ground on a confident detection
        self._pixels = new

    def _update_status(self, k: int) -> None:
        ekf = self.ekfs[k]
        if ekf is None:
            self.statuses[k] = STATUS_LOST
            return
        if ekf.position_std > self.cfg.lost_std_m or ekf.misses > self.cfg.max_misses_coast:
            self.statuses[k] = STATUS_LOST
        elif ekf.misses == 0:
            self.statuses[k] = STATUS_TRACKING
        else:
            self.statuses[k] = STATUS_COASTING

    # -- output -----------------------------------------------------------
    def _emit(self, t, detector_ran, fit, rmse, cur_pts=None, cur_nrm=None,
              cur_tree=None) -> TrackerOutput:
        pts: List[TrackedPoint] = []
        for k, nm in enumerate(LUNG_ORDER):
            ekf = self.ekfs[k]
            if ekf is None or not self.canonical.initialized:
                pts.append(TrackedPoint(nm, np.full(3, np.nan), np.full(3, np.nan),
                                        np.full(2, np.nan), 0.0, STATUS_LOST, np.zeros(3),
                                        float("inf")))
                continue
            p_canon = ekf.position
            cam = transform_points(self._T, p_canon[None])[0]
            if cur_tree is not None:                       # breathing: snap to live surface
                cam, _ = self.canonical.snap_to_surface(cam, cur_tree, cur_pts, cur_nrm)
            pix = self.intr.project(cam[None])[0] if cam[2] > 1e-3 else np.full(2, np.nan)
            world = (transform_points(self.extrinsics, cam[None])[0]
                     if self.extrinsics is not None else None)
            conf = float(1.0 / (1.0 + 50.0 * ekf.position_std))
            pts.append(TrackedPoint(nm, p_canon, cam, pix, conf, self.statuses[k],
                                    ekf.velocity, ekf.position_std, world))
        return TrackerOutput(
            t=t, frame_index=self._frame, initialized=self.canonical.initialized, points=pts,
            T_cam_canon=self._T.copy() if self.canonical.initialized else None,
            icp_fitness=fit, icp_rmse=rmse, detector_ran=detector_ran,
            detector_age_s=t - self._last_detect_t if self._last_detect_t > -1e8 else float("inf"),
            n_tracking=sum(s == STATUS_TRACKING for s in self.statuses),
            n_lost=sum(s == STATUS_LOST for s in self.statuses),
            vo_fitness=self._vo_fit, respiratory_offset_m=self._resp_off,
            respiratory_amplitude_m=self._resp_amp)
