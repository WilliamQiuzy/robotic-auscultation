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
import math
from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

from .detector import LUNG_ORDER
from .geometry import (Intrinsics, body_band_mask, normals_from_pointmap, pointmap_from_depth,
                       sample_depth_m, valid_points)
from .production_geometry_gate import ProductionGeometryGate

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None

STATUS_INIT, STATUS_TRACKING, STATUS_COASTING, STATUS_LOST = "INIT", "TRACKING", "COASTING", "LOST"
TRACK_STATE_COLD_INIT = "COLD_INIT"
TRACK_STATE_ACTIVE = "ACTIVE"
TRACK_STATE_DORMANT = "DORMANT"
TRACK_STATE_REACQUIRING = "REACQUIRING"
REASON_NO_BACK = "NO_BACK"
REASON_INITIALIZING = "INITIALIZING"
REASON_TRACKING = "TRACKING"
REASON_LOST = "LOST"
_RAW_DEPTH_ASSOCIATION_RADIUS_PX = 2  # sample_depth_m's raw-depth lift is a 5x5 footprint
_MATHEMATICAL_SIM3_SUPPORT = 4


@dataclass
class ModelConfig:
    init_frames: int = 12          # confident frames to build the canonical template
    min_conf: float = 0.3          # min detector frame-confidence to use a detection
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
    robot_mask_dilation_px: int = 0  # declared guard band around an aligned robot instance mask
    # Default OFF: preserve a previously trusted body-frame canonical through a
    # long robot occlusion, but publish no camera-frame pose until a gated
    # multi-frame reacquisition succeeds. This remains a staging candidate until
    # its robot+tool mask provider, timing and hand-eye contract are commissioned.
    dormant_reacquisition: bool = False
    reacquire_frames: int = 5
    reacquire_median_resid_m: float = 0.008
    reacquire_max_resid_m: float = 0.015
    reacquire_scale_relative: float = 0.02
    reacquire_translation_step_m: float = 0.010
    reacquire_rotation_step_deg: float = 3.0
    reacquire_scale_step_relative: float = 0.005
    reacquire_absolute_translation_m: float = 0.05
    reacquire_absolute_rotation_deg: float = 15.0
    reacquire_max_point_displacement_m: float = 0.05
    reacquire_max_gap_s: float = 0.2
    # ── perturbation-recovery honesty (2026-08-07 audit) ─────────────────────
    # Snap-vs-blend: the pose EMA exists to kill >2 Hz jitter, not to low-pass a
    # genuine pose change. After a gap the EMA would CRAWL from the held pose to
    # the new fit over ~1/pose_ema frames, publishing confidently-labelled points
    # tens of mm wrong. If the median point displacement between held pose and
    # fresh fit exceeds this, adopt the fresh fit outright.
    pose_snap_m: float = 0.010
    # Debounce the detector-internal depth-plausibility reset: a single-frame
    # flicker (highest risk during patient oscillation) must not wipe the whole
    # track + canonical template. Reset only after this many CONSECUTIVE
    # no-back frames; during the debounce the frame is treated as a miss.
    no_back_reset_frames: int = 3
    # Individual-LOST re-admission: audit showed single points (R3 97% / R2 84%
    # absent) staying LOST for many seconds while the constellation pose stayed
    # valid. Re-admit a LOST point after a consecutive streak of observations
    # that agree with its pose-predicted anatomical position.
    readmit_gate_m: float = 0.03
    readmit_frames: int = 3

    def __post_init__(self) -> None:
        def _real_positive(value) -> bool:
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                and value > 0.0
            )

        if type(self.robot_mask_dilation_px) is not int or self.robot_mask_dilation_px < 0:
            raise ValueError("robot_mask_dilation_px must be a nonnegative int")
        if type(self.dormant_reacquisition) is not bool:
            raise ValueError("dormant_reacquisition must be a bool")
        if type(self.reacquire_frames) is not int or self.reacquire_frames < 5:
            raise ValueError("reacquire_frames must be an int >= 5")
        if not _real_positive(self.reacquire_max_gap_s):
            raise ValueError("reacquire_max_gap_s must be finite and positive")
        if not _real_positive(self.pose_snap_m):
            raise ValueError("pose_snap_m must be finite and positive")
        if type(self.no_back_reset_frames) is not int or self.no_back_reset_frames < 1:
            raise ValueError("no_back_reset_frames must be an int >= 1")
        if not _real_positive(self.readmit_gate_m):
            raise ValueError("readmit_gate_m must be finite and positive")
        if type(self.readmit_frames) is not int or self.readmit_frames < 1:
            raise ValueError("readmit_frames must be an int >= 1")
        positive_reacquisition_limits = (
            self.reacquire_median_resid_m,
            self.reacquire_max_resid_m,
            self.reacquire_scale_relative,
            self.reacquire_translation_step_m,
            self.reacquire_rotation_step_deg,
            self.reacquire_scale_step_relative,
            self.reacquire_absolute_translation_m,
            self.reacquire_absolute_rotation_deg,
            self.reacquire_max_point_displacement_m,
        )
        if not all(_real_positive(value) for value in positive_reacquisition_limits):
            raise ValueError("reacquisition limits must be finite and positive")
        if self.reacquire_median_resid_m > self.reacquire_max_resid_m:
            raise ValueError("reacquire median residual cannot exceed max residual")


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
    xyz_patient: Optional[np.ndarray] = None
    measurement_residual_m: float = float("inf")
    robot_occluded: bool = False
    direct_geometry_accepted: bool = False


@dataclass
class MTOutput:
    t: float
    initialized: bool
    points: List[MTPoint]
    pose_fit_mm: float = 0.0
    n_tracking: int = 0
    n_snapped: int = 0             # of 10 points: how many got a surface-plane snap this frame
    n_measured: int = 0           # of 10 points: how many were actually observed (not extrapolated)
    back_present: bool = False
    detector_confidence: float = 0.0
    reason: str = REASON_NO_BACK
    tracking_state: str = TRACK_STATE_COLD_INIT
    pose_valid: bool = False
    canonical_retained: bool = False
    pose_measured_this_frame: bool = False
    pose_support_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(len(LUNG_ORDER), dtype=np.bool_)
    )
    pose_geometry_accepted: bool = False
    pose_uncertainty_bound_m: float = float("inf")
    sim3_rotation_camera_from_patient: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    sim3_scale: float = 1.0
    sim3_translation_camera_from_patient_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    # DISPLAY-ONLY metadata (2026-08-29): the detector's structure-fitted
    # full-frame pixels for all 10 landmarks from THIS frame's accepted
    # detection (None when no detection was accepted). Never used by the
    # occlusion/publication chain — it lets the operator console render
    # never-initialized (e.g. robot-band-withheld) points in a distinct
    # unverified style instead of hiding anatomy from a human reader.
    detector_pixels: Optional[np.ndarray] = None


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

    def __init__(
        self,
        detector,
        intrinsics: Intrinsics,
        config: Optional[ModelConfig] = None,
        extrinsics: Optional[np.ndarray] = None,
        *,
        geometry_gate: Optional[ProductionGeometryGate] = None,
    ):
        self.detector = detector
        self.intr = intrinsics
        self.cfg = config or ModelConfig()
        self.extrinsics = extrinsics                        # 4x4 camera→base (robot frame), optional
        if geometry_gate is not None and not isinstance(
            geometry_gate, ProductionGeometryGate
        ):
            raise TypeError("geometry_gate must be ProductionGeometryGate")
        self.geometry_gate = geometry_gate
        self.reset()

    def reset(self, reason: str = REASON_NO_BACK) -> None:
        """Forget the current patient completely; no stale coasted points survive a reset."""
        self.canonical: Optional[np.ndarray] = None        # (10,3) body-frame template
        self.residual = np.zeros((len(LUNG_ORDER), 3))      # (10,3) body-frame slow deformation
        self.R = np.eye(3); self.s = 1.0; self.t = np.zeros(3)
        self._have_pose = False
        self._init_obs: List[np.ndarray] = []
        # A point that was hidden throughout global initialization has no safe
        # canonical coordinate yet.  Bootstrap it only after a new consecutive
        # streak is observable through the already-established shared pose.
        self._late_init_obs: List[List[np.ndarray]] = [
            [] for _ in range(len(LUNG_ORDER))
        ]
        self._misses = 0
        self._miss_pp = np.zeros(len(LUNG_ORDER), dtype=int)  # per-point frames-since-measured
        self._initialized = False
        self._cur_pts = None                                # body-band cloud (for surface-snap)
        self._cur_tree = None
        self._n_snapped = 0
        self._frame_back_present: Optional[bool] = None
        self._back_present = False
        self._detector_confidence = 0.0
        self._reset_reason = reason
        self._tracking_state = TRACK_STATE_COLD_INIT
        self._dormant_reference_scale = 1.0
        self._dormant_reference_R = np.eye(3)
        self._dormant_reference_translation = np.zeros(3)
        self._dormant_reference_timestamp: Optional[float] = None
        self._dormant_entry_timestamp: Optional[float] = None
        self._dormant_last_timestamp: Optional[float] = None
        self._last_trusted_timestamp: Optional[float] = None
        self._reacquire_proposals = []
        self._frame_robot_occluded = np.zeros(len(LUNG_ORDER), dtype=np.bool_)
        self._frame_pose_measured = False
        self._frame_pose_support = np.zeros(len(LUNG_ORDER), dtype=np.bool_)
        self._frame_pose_geometry_accepted = False
        self._frame_pose_uncertainty_bound_m = float("inf")
        self._frame_measurement_residual_m = np.full(
            len(LUNG_ORDER), np.inf, dtype=np.float64,
        )
        self._no_back_streak = 0
        self._readmit_obs: List[List[np.ndarray]] = [
            [] for _ in range(len(LUNG_ORDER))
        ]

    # -- body cloud + ray surface-snap (Step 1: stable depth at the source) ------
    def _expanded_robot_mask(self, robot_mask: np.ndarray) -> np.ndarray:
        """Return the declared square-dilated aligned robot mask."""

        dilation = self.cfg.robot_mask_dilation_px
        if dilation == 0 or not robot_mask.any():
            return robot_mask
        height, width = robot_mask.shape
        if dilation >= max(height - 1, width - 1):
            return np.full(robot_mask.shape, bool(robot_mask.any()), dtype=bool)
        # Integral-image square dilation is O(H*W), independent of radius.  A
        # nested shift/OR implementation becomes a worker-level denial of
        # service for large but otherwise valid configuration values.
        integral = np.pad(
            robot_mask.astype(np.uint32, copy=False),
            ((1, 0), (1, 0)),
            mode="constant",
        )
        integral = np.cumsum(integral, axis=0, dtype=np.uint64)
        integral = np.cumsum(integral, axis=1, dtype=np.uint64)
        rows = np.arange(height)
        cols = np.arange(width)
        y0 = np.maximum(rows - dilation, 0)
        y1 = np.minimum(rows + dilation + 1, height)
        x0 = np.maximum(cols - dilation, 0)
        x1 = np.minimum(cols + dilation + 1, width)
        counts = (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )
        return counts > 0

    def _build_cloud(self, depth, robot_mask=None):
        if not self.cfg.surface_snap or cKDTree is None:
            self._cur_pts = self._cur_tree = None
            return
        P = pointmap_from_depth(depth, self.intr, stride=self.cfg.icp_stride, max_m=self.cfg.max_depth_m)
        if robot_mask is not None:
            # The raw 5x5 association gate is narrower than the metric
            # surface-snap neighbourhood.  Remove every declared robot sample
            # from the KD-tree too, otherwise a nearby robot plane can pull an
            # otherwise eligible patient ray by several millimetres.
            self._validate_robot_mask(robot_mask, depth)
            cloud_robot_mask = self._expanded_robot_mask(robot_mask)[
                :: self.cfg.icp_stride, :: self.cfg.icp_stride
            ]
            if cloud_robot_mask.shape != P.shape[:2]:
                raise ValueError("robot_mask downsample does not align to the snap cloud")
            P[cloud_robot_mask] = np.nan
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

    def _validate_robot_mask(self, robot_mask, depth) -> None:
        """Validate array shape only, not provider provenance.

        Production callers must separately prove that this is the frontmost,
        depth-aware robot+tool mask for the same aligned RGB-D frame and the
        correct delayed joint/hand-eye state.  A bool array cannot carry or
        establish that commissioning contract by itself.
        """
        dilation = self.cfg.robot_mask_dilation_px
        if type(dilation) is not int or dilation < 0:
            raise ValueError("robot_mask_dilation_px must be a nonnegative int")
        if not isinstance(robot_mask, np.ndarray) or robot_mask.dtype != np.bool_ \
                or robot_mask.ndim != 2:
            raise ValueError("robot_mask must be a 2-D bool numpy array aligned to depth")
        if depth is None or np.ndim(depth) != 2 or robot_mask.shape != np.shape(depth):
            raise ValueError("robot_mask shape must exactly match the aligned depth frame")

    def _reject_robot_mask_associations(self, uv, depth, robot_mask):
        """NaN detector points whose 5x5 raw-depth association touches the robot mask."""
        filtered = np.asarray(uv, dtype=np.float64).copy()
        filtered[self._robot_mask_association_flags(filtered, depth, robot_mask)] = np.nan
        return filtered

    def _robot_mask_association_flags(self, uv, depth, robot_mask):
        """Return which detector-depth footprints intersect the aligned robot mask."""
        self._validate_robot_mask(robot_mask, depth)
        pixels = np.asarray(uv, dtype=np.float64)
        blocked = np.zeros(len(pixels), dtype=np.bool_)
        height, width = depth.shape
        # Checking the original mask over an expanded footprint is exactly equivalent to
        # checking the 5x5 footprint against a square-dilated mask, without a full-frame copy.
        radius = _RAW_DEPTH_ASSOCIATION_RADIUS_PX + self.cfg.robot_mask_dilation_px
        for index, (u, v) in enumerate(pixels):
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            ui, vi = int(round(float(u))), int(round(float(v)))
            if not (0 <= ui < width and 0 <= vi < height):
                continue
            y0, y1 = max(0, vi - radius), min(height, vi + radius + 1)
            x0, x1 = max(0, ui - radius), min(width, ui + radius + 1)
            if robot_mask[y0:y1, x0:x1].any():
                blocked[index] = True
        return blocked

    # -- detection → 3D ---------------------------------------------------
    def _detect3d(self, rgb, depth, robot_mask=None):
        det = self.detector.detect(rgb, depth, self.intr)
        self._frame_back_present = None
        self._detector_confidence = 0.0
        if det is None:
            return None
        try:
            confidence = float(det.confidence)
        except (TypeError, ValueError):
            return None
        explicit_presence = getattr(det, "back_present", None)
        if explicit_presence is not None:
            self._frame_back_present = bool(explicit_presence)
        if not np.isfinite(confidence):
            return None
        self._detector_confidence = float(np.clip(confidence, 0.0, 1.0))
        if explicit_presence is False or confidence < self.cfg.min_conf:
            return None
        if explicit_presence is None:
            # Backward-compatible detectors have no geometry gate; an accepted detection is
            # evidence of presence, but production uses NormalsDetector's explicit boolean.
            self._frame_back_present = True
        full = det.crop.det_to_full(det.xy_det)
        self._frame_detector_pixels_full = np.array(full, copy=True)
        if robot_mask is not None:
            self._frame_robot_occluded = self._robot_mask_association_flags(
                full, depth, robot_mask,
            )
            full = self._reject_robot_mask_associations(full, depth, robot_mask)
        if not self.cfg.surface_snap or self._cur_tree is None:
            self._n_snapped = 0
            return self._unproject(full, depth)
        seed_pred = self._pose_apply(self.canonical + self.residual) if self._have_pose else None
        obs, self._n_snapped = self._snap_rays(full, depth, seed_pred)
        return obs                                           # (10,3), NaN where no depth/snap

    # -- canonical build (generalized Procrustes over init frames) --------
    def _align_avg(self, obs_list, ref):
        """Procrustes-align each obs to ref and average; also return each frame's alignment residual."""
        acc = np.zeros_like(ref); cnt = np.zeros(len(ref)); resid = []
        for obs in obs_list:
            m = np.isfinite(ref).all(1) & np.isfinite(obs).all(1)
            if m.sum() < _MATHEMATICAL_SIM3_SUPPORT:
                resid.append(np.inf); continue
            R, s, t = _umeyama(obs[m], ref[m], np.ones(m.sum()))
            al = (s * (R @ obs.T).T + t); ok = np.isfinite(al).all(1)
            acc[ok] += al[ok]; cnt[ok] += 1
            resid.append(float(np.median(np.linalg.norm(al[m] - ref[m], axis=1))))
        canon = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), ref)
        return canon, np.array(resid)

    def _build_canonical(self, fully_supported: np.ndarray) -> bool:
        obs = self._init_obs
        canon, _ = self._align_avg(obs, obs[0])                 # first pass: rough mean
        canon, res = self._align_avg(obs, canon)               # align to mean → per-frame residual
        # drop inconsistent init frames (e.g. captured mid-motion/turn) so they don't blur the template
        keep = res < (np.median(res) + 2.0 * np.median(np.abs(res - np.median(res))) + 1e-6)
        if keep.sum() >= max(3, len(obs) // 2):
            canon, _ = self._align_avg([o for o, k in zip(obs, keep) if k], canon)
        support = np.asarray(fully_supported, dtype=bool)
        if support.shape != (len(LUNG_ORDER),):
            raise ValueError("canonical support mask shape changed")
        # Align every initialization frame back into the candidate body frame,
        # then reject per-point histories that are not stable.  A point must be
        # present for the entire global streak; one late depth sample must never
        # bypass the delayed per-point bootstrap below.
        aligned_frames = []
        for frame in obs:
            fit_mask = support & np.isfinite(frame).all(1) & np.isfinite(canon).all(1)
            if int(fit_mask.sum()) < _MATHEMATICAL_SIM3_SUPPORT:
                self.canonical = None
                self._have_pose = False
                return False
            align_R, align_s, align_t = _umeyama(
                frame[fit_mask], canon[fit_mask], np.ones(int(fit_mask.sum()))
            )
            aligned_frames.append(align_s * (align_R @ frame.T).T + align_t)
        aligned = np.asarray(aligned_frames, dtype=np.float64)
        stable = np.zeros(len(LUNG_ORDER), dtype=bool)
        supported_values = aligned[:, support, :]
        centre = np.median(supported_values, axis=0)
        spread = np.linalg.norm(supported_values - centre[None], axis=2)
        stable[support] = (
            np.all(np.isfinite(supported_values), axis=(0, 2))
            & (np.max(spread, axis=0) <= self.cfg.huber_m)
        )
        canon[~stable] = np.nan
        if int(np.count_nonzero(stable)) < _MATHEMATICAL_SIM3_SUPPORT:
            self.canonical = None
            self._have_pose = False
            return False
        self.canonical = canon
        self.R, self.s, self.t = np.eye(3), 1.0, np.zeros(3)
        self._have_pose = True
        # ``_miss_pp == 0`` means measured on this exact output frame.  Preserve
        # that contract for partially observed initialization instead of
        # falsely labelling never-seen NaN canonical points as TRACKING.
        for index in range(len(LUNG_ORDER)):
            if not np.isfinite(self.canonical[index]).all():
                self._miss_pp[index] = self.cfg.coast_max + 1
                continue
            trailing_misses = 0
            for frame in reversed(obs):
                if np.isfinite(frame[index]).all():
                    break
                trailing_misses += 1
            self._miss_pp[index] = trailing_misses
        return True

    def _bootstrap_missing_canonical(self, obs: np.ndarray) -> np.ndarray:
        """Initialize never-seen points from a stable body-frame observation streak.

        The shared pose is estimated only from already-known canonical points.
        A newly revealed point remains LOST until ``init_frames`` consecutive
        body-frame samples agree within the normal robust-fit scale.  This avoids
        both the old permanent-NaN identity loss and a one-frame robot/depth
        transient becoming a canonical anatomical point.
        """

        recovered = np.zeros(len(LUNG_ORDER), dtype=bool)
        obs_body = self._pose_inv(obs)
        required = max(1, int(self.cfg.init_frames))
        for index in range(len(LUNG_ORDER)):
            if np.isfinite(self.canonical[index]).all():
                self._late_init_obs[index].clear()
                continue
            sample = obs_body[index]
            if not np.isfinite(sample).all():
                self._late_init_obs[index].clear()
                continue
            history = self._late_init_obs[index]
            history.append(sample.copy())
            if len(history) > required:
                del history[:-required]
            if len(history) < required:
                continue
            values = np.asarray(history, dtype=np.float64)
            centre = np.median(values, axis=0)
            spread = np.linalg.norm(values - centre, axis=1)
            if float(np.max(spread)) > self.cfg.huber_m:
                # Require a fresh consecutive streak after an inconsistent
                # reveal; keep only the current sample as the new start.
                history[:] = [sample.copy()]
                continue
            self.canonical[index] = values.mean(axis=0)
            self.residual[index] = 0.0
            history.clear()
            recovered[index] = True
        return recovered

    def _readmit_lost_points(self, obs: np.ndarray) -> np.ndarray:
        """Re-admit an individually-LOST point whose observations have returned.

        Audit evidence (2026-08-07): single points stayed LOST for many seconds
        (R3 absent 97% of a static session) while the constellation pose stayed
        valid — nothing re-admitted a point once ``_miss_pp`` exceeded
        ``coast_max`` unless it happened to pass the tight inlier band. Require a
        consecutive streak of observations that agree with the pose-predicted
        anatomical position within ``readmit_gate_m``; on success refresh the
        slow residual from the streak so drift during the absence is absorbed
        instead of blocking the inlier gate forever.
        """
        readmitted = np.zeros(len(LUNG_ORDER), dtype=bool)
        lost = self._miss_pp > int(self.cfg.coast_max)
        predicted = self._pose_apply(self.canonical + self.residual)
        required = max(1, int(self.cfg.readmit_frames))
        for index in range(len(LUNG_ORDER)):
            history = self._readmit_obs[index]
            if not lost[index] or not np.isfinite(self.canonical[index]).all():
                history.clear()
                continue
            sample = obs[index]
            if (
                not np.isfinite(sample).all()
                or not np.isfinite(predicted[index]).all()
                or float(np.linalg.norm(sample - predicted[index]))
                > self.cfg.readmit_gate_m
            ):
                history.clear()
                continue
            history.append(sample.copy())
            if len(history) > required:
                del history[:-required]
            if len(history) < required:
                continue
            values = np.asarray(history, dtype=np.float64)
            centre = np.median(values, axis=0)
            spread = np.linalg.norm(values - centre, axis=1)
            if float(np.max(spread)) > self.cfg.huber_m:
                history[:] = [sample.copy()]
                continue
            # Absorb drift-during-absence into the (clamped) slow residual so the
            # point publishes at the observed anatomy immediately.
            body = self._pose_inv(values.mean(axis=0)[None, :])[0]
            offset = body - self.canonical[index]
            norm = float(np.linalg.norm(offset))
            if norm > self.cfg.max_resid_m:
                offset = offset / max(norm, 1e-9) * self.cfg.max_resid_m
            self.residual[index] = offset
            history.clear()
            readmitted[index] = True
        return readmitted

    # -- robust pose fit (IRLS) -------------------------------------------
    def _evaluate_pose_geometry(self, source, destination, target_template):
        if self.geometry_gate is None:
            return None
        source = np.asarray(source, dtype=np.float64)
        destination = np.asarray(destination, dtype=np.float64)
        targets = np.asarray(target_template, dtype=np.float64)
        targets = targets[np.isfinite(targets).all(axis=1)]
        if len(targets) == 0:
            return None
        # The point detector cannot authorize its own z samples.  A production
        # decision requires a separately bound same-frame RGB-D/back/robot mask
        # and independent patient-model evidence object.  Until that provider
        # is wired at the frame-bundle boundary, fail closed explicitly.
        return self.geometry_gate.evaluate(
            reference_points_m=source,
            observed_points_m=destination,
            target_reference_points_m=targets,
            evidence=None,
        )

    def _fit_pose(self, obs, template=None):
        source_template = self.canonical if template is None else np.asarray(template, dtype=np.float64)
        if source_template is None or source_template.shape != (len(LUNG_ORDER), 3):
            return None
        m = np.isfinite(obs).all(1) & np.isfinite(source_template).all(1)
        if m.sum() < _MATHEMATICAL_SIM3_SUPPORT:
            return None
        src = source_template[m]; dst = obs[m]
        w = np.ones(m.sum())
        R = s = t = None
        for _ in range(self.cfg.irls_iters):
            R, s, t = _umeyama(src, dst, w)
            r = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
            w = 1.0 / np.maximum(r / self.cfg.huber_m, 1.0)     # Huber-ish
        # Least-trimmed-squares hardening: with 10 landmarks, retain the best 70% (never fewer
        # than the mathematical Sim(3) floor) and refit repeatedly. This breaks coherent multi-point occluder
        # clusters that can still pull a soft IRLS solution far enough to reject the true points.
        h = min(
            len(src),
            max(_MATHEMATICAL_SIM3_SUPPORT, int(round(0.7 * len(src)))),
        )
        for _ in range(5):
            r = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
            keep = np.argsort(r, kind="stable")[:h]
            R, s, t = _umeyama(src[keep], dst[keep], np.ones(h))
        # hard-inlier refit: drop residual outliers entirely so they can't pull the pose (a few
        # transient detector glitches must not corrupt the other points)
        inl = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1) < self.cfg.inlier_huber_mult * self.cfg.huber_m
        if int(inl.sum()) >= _MATHEMATICAL_SIM3_SUPPORT:
            R, s, t = _umeyama(src[inl], dst[inl], np.ones(int(inl.sum())))
        resid = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
        fit = float(np.median(resid))
        # a point is MEASURED iff it had a valid observation AND is a fit inlier (not an outlier
        # the IRLS down-weighted). Points outside `m` were never observed → extrapolated → not measured.
        measured = np.zeros(len(LUNG_ORDER), dtype=bool)
        measured[np.flatnonzero(m)] = resid < self.cfg.inlier_huber_mult * self.cfg.huber_m
        geometry_decision = self._evaluate_pose_geometry(
            src,
            dst,
            source_template,
        )
        return R, s, t, measured, fit, geometry_decision

    def _has_reacquisition_coverage(self, mask: np.ndarray) -> bool:
        """Apply only the mathematical floor; geometry quality is gated separately."""

        supported = np.asarray(mask, dtype=bool)
        if supported.shape != (len(LUNG_ORDER),):
            return False
        return bool(int(supported.sum()) >= _MATHEMATICAL_SIM3_SUPPORT)

    @staticmethod
    def _rotation_delta_deg(previous: np.ndarray, current: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(previous.T @ current) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _reacquisition_proposal(self, obs):
        """Return one fail-closed pose proposal without mutating the live track."""

        if obs is None or self.canonical is None:
            return None
        # This must be the exact template `_emit` will publish after recovery.
        # Gating canonical alone and then adding a retained residual at output
        # time can move a supposedly accepted point beyond the 15 mm contract.
        retained_template = self.canonical + self.residual
        valid = np.isfinite(obs).all(1) & np.isfinite(retained_template).all(1)
        if not self._has_reacquisition_coverage(valid):
            return None
        fitres = self._fit_pose(obs, template=retained_template)
        if fitres is None:
            return None
        R, scale, translation, _measured, fit, geometry_decision = fitres
        if geometry_decision is not None and not geometry_decision.accepted:
            return None
        if (
            not np.isfinite(R).all()
            or not np.isfinite(scale)
            or scale <= 0.0
            or not np.isfinite(translation).all()
        ):
            return None
        predicted = scale * (R @ retained_template.T).T + translation
        residual = np.linalg.norm(obs - predicted, axis=1)
        if not np.isfinite(residual[valid]).all():
            return None
        if float(np.median(residual[valid])) > self.cfg.reacquire_median_resid_m:
            return None
        accepted = valid & (residual <= self.cfg.reacquire_max_resid_m)
        if not self._has_reacquisition_coverage(accepted):
            return None
        if float(np.max(residual[accepted])) > self.cfg.reacquire_max_resid_m:
            return None
        reference_scale = max(float(self._dormant_reference_scale), 1e-9)
        if abs(scale / reference_scale - 1.0) > self.cfg.reacquire_scale_relative:
            return None
        if (
            np.linalg.norm(translation - self._dormant_reference_translation)
            > self.cfg.reacquire_absolute_translation_m
            or self._rotation_delta_deg(self._dormant_reference_R, R)
            > self.cfg.reacquire_absolute_rotation_deg
        ):
            return None
        finite_template = np.isfinite(retained_template).all(1)
        if not finite_template.any():
            return None
        retained_finite = retained_template[finite_template]
        reference_points = (
            self._dormant_reference_scale
            * (self._dormant_reference_R @ retained_finite.T).T
            + self._dormant_reference_translation
        )
        proposal_points = scale * (R @ retained_finite.T).T + translation
        point_displacement = np.linalg.norm(proposal_points - reference_points, axis=1)
        if (
            not np.isfinite(point_displacement).all()
            or float(np.max(point_displacement))
            > self.cfg.reacquire_max_point_displacement_m
        ):
            return None
        return R, float(scale), translation, accepted, float(fit)

    def _enter_dormant(self, timestamp: float) -> None:
        """Invalidate camera pose while retaining the immutable body-frame patient model."""

        self._dormant_reference_scale = float(self.s)
        self._dormant_reference_R = self.R.copy()
        self._dormant_reference_translation = self.t.copy()
        self._dormant_reference_timestamp = self._last_trusted_timestamp
        entry_timestamp = float(timestamp)
        self._dormant_entry_timestamp = (
            entry_timestamp if np.isfinite(entry_timestamp) else None
        )
        self._dormant_last_timestamp = self._dormant_entry_timestamp
        self._initialized = False
        self._have_pose = False
        self._tracking_state = TRACK_STATE_DORMANT
        self._reacquire_proposals.clear()
        for history in self._late_init_obs:
            history.clear()

    def _emit_dormant(self, t: float) -> MTOutput:
        points = [
            MTPoint(
                name,
                np.full(3, np.nan),
                np.full(2, np.nan),
                STATUS_LOST,
                measured=False,
                position_std=float("inf"),
            )
            for name in LUNG_ORDER
        ]
        return MTOutput(
            t=t,
            initialized=False,
            points=points,
            back_present=self._back_present,
            detector_confidence=self._detector_confidence,
            reason=REASON_LOST,
            tracking_state=self._tracking_state,
            pose_valid=False,
            canonical_retained=self.canonical is not None,
        )

    def _process_dormant(self, obs, t: float) -> MTOutput:
        timestamp = float(t)
        previous_timestamp = self._dormant_last_timestamp
        if not np.isfinite(timestamp) or previous_timestamp is None:
            self._reacquire_proposals.clear()
            self._tracking_state = TRACK_STATE_DORMANT
            if np.isfinite(timestamp):
                self._dormant_last_timestamp = timestamp
            return self._emit_dormant(t)
        gap = timestamp - previous_timestamp
        if gap <= 0.0 or gap > self.cfg.reacquire_max_gap_s:
            self._reacquire_proposals.clear()
            self._tracking_state = TRACK_STATE_DORMANT
            # A forward but stale frame establishes a new time baseline; it is
            # never counted as evidence. Duplicate/backward time cannot move
            # the baseline and therefore cannot be replayed into a streak.
            if gap > 0.0:
                self._dormant_last_timestamp = timestamp
            return self._emit_dormant(t)
        self._dormant_last_timestamp = timestamp

        proposal = self._reacquisition_proposal(obs)
        if proposal is None:
            self._reacquire_proposals.clear()
            self._tracking_state = TRACK_STATE_DORMANT
            return self._emit_dormant(t)

        R, scale, translation, accepted, fit = proposal
        if self._reacquire_proposals:
            (
                _proposal_timestamp,
                previous_R,
                previous_scale,
                previous_translation,
                _accepted,
                _fit,
            ) = self._reacquire_proposals[-1]
            if (
                np.linalg.norm(translation - previous_translation)
                > self.cfg.reacquire_translation_step_m
                or self._rotation_delta_deg(previous_R, R)
                > self.cfg.reacquire_rotation_step_deg
                or abs(scale / max(previous_scale, 1e-9) - 1.0)
                > self.cfg.reacquire_scale_step_relative
            ):
                self._reacquire_proposals.clear()
                self._tracking_state = TRACK_STATE_DORMANT
                return self._emit_dormant(t)

        self._reacquire_proposals.append((timestamp, *proposal))
        self._tracking_state = TRACK_STATE_REACQUIRING
        if len(self._reacquire_proposals) < self.cfg.reacquire_frames:
            return self._emit_dormant(t)

        # Commit the latest already-gated pose directly. In particular, do not
        # interpolate from the stale pre-occlusion camera pose.
        self.R = R.copy()
        self.s = float(scale)
        self.t = translation.copy()
        self._have_pose = True
        self._initialized = True
        self._tracking_state = TRACK_STATE_ACTIVE
        self._misses = 0
        self._miss_pp = np.where(accepted, 0, self.cfg.coast_max + 1)
        self._last_trusted_timestamp = timestamp
        self._reacquire_proposals.clear()
        return self._emit(t, fit_mm=fit * 1000.0)

    def _smooth_pose(self, R, s, t):
        a = self.cfg.pose_ema
        # Recovery snap: a genuine pose change (post-gap/perturbation) must not be
        # low-passed into a multi-frame crawl of confidently-labelled wrong points.
        if self.canonical is not None:
            valid = np.isfinite(self.canonical).all(1)
            if np.any(valid):
                held = self._pose_apply(self.canonical[valid])
                fresh = s * (R @ self.canonical[valid].T).T + t
                displacement = float(
                    np.median(np.linalg.norm(held - fresh, axis=1))
                )
                if displacement > self.cfg.pose_snap_m:
                    self.R = np.asarray(R, dtype=np.float64).copy()
                    self.s = float(s)
                    self.t = np.asarray(t, dtype=np.float64).copy()
                    return
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
    def process_persisted_model_input(self, model_input):
        """Run only on arrays reopened from the lossless evidence boundary."""

        # Import locally so the ordinary diagnostic tracker remains usable in
        # installations that do not activate production evidence capture.
        from .production_evidence_capture import PersistedGeometryModelInput

        if not isinstance(model_input, PersistedGeometryModelInput):
            raise TypeError("model_input must be PersistedGeometryModelInput")
        reloaded = model_input.reload_for_inference()
        return self.process(
            reloaded.rgb,
            reloaded.depth,
            t=model_input.camera_t,
            depth_scale=model_input.depth_scale,
            robot_mask=reloaded.robot_mask,
        )

    def process(self, rgb, depth, t: float = 0.0, depth_scale: Optional[float] = None,
                *, robot_mask=None) -> MTOutput:
        output = self._process_frame(
            rgb, depth, t=t, depth_scale=depth_scale, robot_mask=robot_mask,
        )
        pixels = getattr(self, "_frame_detector_pixels_full", None)
        output.detector_pixels = (
            None if pixels is None else np.array(pixels, copy=True)
        )
        return output

    def _process_frame(self, rgb, depth, t: float = 0.0,
                       depth_scale: Optional[float] = None,
                       *, robot_mask=None) -> MTOutput:
        self._frame_detector_pixels_full = None
        self._frame_robot_occluded.fill(False)
        self._frame_pose_measured = False
        self._frame_pose_support.fill(False)
        self._frame_pose_geometry_accepted = False
        self._frame_pose_uncertainty_bound_m = float("inf")
        self._frame_measurement_residual_m.fill(float("inf"))
        if robot_mask is not None:
            self._validate_robot_mask(robot_mask, depth)
        if depth_scale is not None and abs(depth_scale - self.intr.depth_scale) > 1e-9:
            import dataclasses
            self.intr = dataclasses.replace(self.intr, depth_scale=depth_scale)
        self._n_snapped = 0
        self._frame_back_present = None
        if depth is not None:
            self._build_cloud(
                depth, robot_mask=robot_mask,
            )                                                # body cloud for ray surface-snap
        obs = None if (rgb is None or depth is None) else self._detect3d(
            rgb, depth, robot_mask=robot_mask,
        )

        if self._frame_back_present is False:
            # Debounce: a single-frame depth-plausibility flicker must not wipe the
            # canonical template (a reset also rebases the body frame). Persist the
            # streak; within the debounce window treat the frame as a plain miss.
            self._no_back_streak += 1
            if self._no_back_streak >= int(self.cfg.no_back_reset_frames):
                self.reset(REASON_NO_BACK)
                return self._emit(t, reason=REASON_NO_BACK)
            obs = None
        else:
            self._no_back_streak = 0
        if self._frame_back_present is True:
            self._back_present = True

        if (
            self.cfg.dormant_reacquisition
            and self._tracking_state in (TRACK_STATE_DORMANT, TRACK_STATE_REACQUIRING)
        ):
            return self._process_dormant(obs, t)

        if not self._initialized:
            if (
                obs is not None
                and np.isfinite(obs).all(1).sum()
                >= _MATHEMATICAL_SIM3_SUPPORT
            ):
                self._init_obs.append(obs)
                self._reset_reason = REASON_INITIALIZING
            else:
                # Initialization is a streak, never a bag of sparse good frames accumulated
                # across disappearances/dropouts.
                self._init_obs.clear()
                if self._frame_back_present is not True:
                    self._back_present = False
            required = max(1, int(self.cfg.init_frames))
            if len(self._init_obs) >= required:
                window = self._init_obs[-required:]
                fully_supported = np.all(
                    np.asarray(
                        [np.isfinite(frame).all(1) for frame in window],
                        dtype=bool,
                    ),
                    axis=0,
                )
                self._initialized = bool(
                    int(np.count_nonzero(fully_supported))
                    >= _MATHEMATICAL_SIM3_SUPPORT
                    and self._build_canonical(fully_supported)
                )
                if self._initialized:
                    self._tracking_state = TRACK_STATE_ACTIVE
                    supported = np.isfinite(self.canonical).all(1)
                    self._frame_pose_measured = True
                    self._frame_pose_support = supported.astype(np.bool_, copy=True)
                    latest = window[-1]
                    valid_geometry = (
                        np.isfinite(self.canonical).all(1)
                        & np.isfinite(latest).all(1)
                    )
                    geometry_decision = self._evaluate_pose_geometry(
                        self.canonical[valid_geometry],
                        latest[valid_geometry],
                        self.canonical,
                    )
                    self._frame_pose_geometry_accepted = bool(
                        geometry_decision is not None
                        and geometry_decision.accepted
                    )
                    self._frame_pose_uncertainty_bound_m = (
                        float(geometry_decision.max_target_error_p95_m)
                        if self._frame_pose_geometry_accepted
                        else float("inf")
                    )
                    self._frame_measurement_residual_m[supported] = 0.0
                    trusted_timestamp = float(t)
                    self._last_trusted_timestamp = (
                        trusted_timestamp if np.isfinite(trusted_timestamp) else None
                    )
                if not self._initialized:
                    # Slide the consecutive global window by one frame; never
                    # accumulate sparse support across a failed interval.
                    self._init_obs = (
                        []
                        if required == 1
                        else self._init_obs[-(required - 1) :]
                    )
            return self._emit(t, init_only=True)

        fitres = self._fit_pose(obs) if obs is not None else None
        if fitres is None:
            for history in self._late_init_obs:
                history.clear()
            self._misses += 1
            self._miss_pp += 1                               # no detection → every point extrapolated
            if np.all(self._miss_pp > self.cfg.coast_max):
                if self.cfg.dormant_reacquisition:
                    self._enter_dormant(t)
                    return self._emit_dormant(t)
                self.reset(REASON_LOST)
                return self._emit(t, reason=REASON_LOST)
            return self._emit(t, fit_mm=0.0)                 # hold last smoothed pose, per-point status
        self._misses = 0
        R, s, tt, measured, fit, geometry_decision = fitres
        valid = (
            np.isfinite(obs).all(1)
            & np.isfinite(self.canonical).all(1)
        )
        predicted = s * (R @ self.canonical.T).T + tt
        self._frame_measurement_residual_m[valid] = np.linalg.norm(
            obs[valid] - predicted[valid], axis=1,
        )
        self._frame_pose_measured = True
        self._frame_pose_support = measured.astype(np.bool_, copy=True)
        self._frame_pose_geometry_accepted = bool(
            geometry_decision is not None and geometry_decision.accepted
        )
        self._frame_pose_uncertainty_bound_m = (
            float(geometry_decision.max_target_error_p95_m)
            if self._frame_pose_geometry_accepted
            else float("inf")
        )
        self._smooth_pose(R, s, tt)
        self._tracking_state = TRACK_STATE_ACTIVE
        trusted_timestamp = float(t)
        self._last_trusted_timestamp = (
            trusted_timestamp if np.isfinite(trusted_timestamp) else None
        )
        measured = measured | self._bootstrap_missing_canonical(obs)
        measured = measured | self._readmit_lost_points(obs)
        self._miss_pp = np.where(measured, 0, self._miss_pp + 1)
        # slow per-point residual in body frame (absorbs non-rigid deformation, clamped). Only update
        # from MEASURED points — never let an extrapolated point feed its own (circular) residual.
        resid_obs_body = self._pose_inv(obs) - self.canonical
        upd = measured & np.isfinite(resid_obs_body).all(1)
        self.residual[upd] = (1 - self.cfg.resid_ema) * self.residual[upd] + self.cfg.resid_ema * resid_obs_body[upd]
        n = np.linalg.norm(self.residual, axis=1, keepdims=True)
        self.residual = np.where(n > self.cfg.max_resid_m, self.residual / np.maximum(n, 1e-9) * self.cfg.max_resid_m, self.residual)
        return self._emit(t, fit_mm=fit * 1000.0)

    def _emit(self, t, fit_mm: float = 0.0, init_only: bool = False,
              reason: Optional[str] = None) -> MTOutput:
        if not self._initialized:
            pts = [MTPoint(nm, np.full(3, np.nan), np.full(2, np.nan), STATUS_INIT) for nm in LUNG_ORDER]
            why = reason or (REASON_INITIALIZING if self._back_present else self._reset_reason)
            return MTOutput(t, False, pts, back_present=self._back_present,
                            detector_confidence=self._detector_confidence, reason=why,
                            tracking_state=self._tracking_state, pose_valid=False,
                            canonical_retained=self.canonical is not None)
        cam = self._pose_apply(self.canonical + self.residual)
        patient = self.canonical + self.residual
        pts = []
        ntr = nmeas = 0
        for k, nm in enumerate(LUNG_ORDER):
            c = cam[k]
            pix = (
                self.intr.project(c[None])[0]
                if np.isfinite(c).all() and c[2] > 1e-3
                else np.full(2, np.nan)
            )
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
            residual_m = float(self._frame_measurement_residual_m[k])
            robot_occluded = bool(self._frame_robot_occluded[k])
            direct_geometry_accepted = bool(
                measured
                and not robot_occluded
                and self._frame_pose_geometry_accepted
                and math.isfinite(residual_m)
                and residual_m
                <= self.geometry_gate.profile.decision_surface.max_residual_m
            )
            pts.append(MTPoint(
                nm,
                c,
                pix,
                st,
                measured=measured,
                position_std=std,
                xyz_world=world,
                xyz_patient=patient[k].copy(),
                measurement_residual_m=residual_m,
                robot_occluded=robot_occluded,
                direct_geometry_accepted=direct_geometry_accepted,
            ))
        return MTOutput(t, True, pts, pose_fit_mm=fit_mm, n_tracking=ntr,
                        n_snapped=self._n_snapped, n_measured=nmeas,
                        back_present=self._back_present,
                        detector_confidence=self._detector_confidence,
                        reason=reason or REASON_TRACKING,
                        tracking_state=TRACK_STATE_ACTIVE, pose_valid=True,
                        canonical_retained=self.canonical is not None,
                        pose_measured_this_frame=self._frame_pose_measured,
                        pose_support_mask=self._frame_pose_support.copy(),
                        pose_geometry_accepted=self._frame_pose_geometry_accepted,
                        pose_uncertainty_bound_m=self._frame_pose_uncertainty_bound_m,
                        sim3_rotation_camera_from_patient=self.R.copy(),
                        sim3_scale=float(self.s),
                        sim3_translation_camera_from_patient_m=self.t.copy())
