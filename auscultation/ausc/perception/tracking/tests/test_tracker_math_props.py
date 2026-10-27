"""Property / invariant / adversarial tests for the pose math in model_tracker.py.

Torch-free: we exercise ``_umeyama`` directly and ``ModelBasedTracker`` with a tiny stub
detector that returns a ``DetectorResult`` carrying a chosen ``xy_det`` (no model checkpoint,
no GPU, no real data). All randomness is seeded (``np.random.default_rng``).

What these cover (per the tracker docstring contract):
  * _umeyama recovers a KNOWN (R,s,t) exactly from clean correspondences
  * _umeyama never returns a reflection (det(R)==+1) even when dst IS a reflection of src
  * _fit_pose IRLS hard-inlier refit: a few gross outliers must NOT corrupt the inlier pose
  * _fit_pose returns None below the Sim(3) support floor of finite correspondences
  * all-coincident (zero-variance) src does not crash and stays finite/sane
  * _build_canonical refuses a window holding a mid-motion frame, then builds a clean template once
    the window has moved past it
  * _smooth_pose EMA stays a valid similarity (orthonormal R, bounded s/t) over many steps
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.tracking.model_tracker import (  # noqa: E402
    _MATHEMATICAL_SIM3_SUPPORT, ModelBasedTracker, ModelConfig, _umeyama)
from ausc.perception.tracking.detector import DetectorResult, LUNG_ORDER  # noqa: E402
from ausc.perception.tracking.geometry import Intrinsics, CropTransform  # noqa: E402

N = len(LUNG_ORDER)  # 10


# ── helpers ───────────────────────────────────────────────────────────────────
def _rand_rot(rng):
    """A uniformly-drawn proper rotation (det=+1) via QR, sign-fixed."""
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def _intr():
    return Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480, depth_scale=1.0)


def _tracker_with_canonical(canon, cfg=None):
    """A tracker with ``canonical`` injected so _fit_pose can run without a detector."""
    tr = ModelBasedTracker(detector=None, intrinsics=_intr(), config=cfg or ModelConfig())
    tr.canonical = canon.copy()
    return tr


class _StubDetector:
    """Returns a DetectorResult with a caller-supplied xy_det in FULL-frame pixels (the crop is
    identity so det_to_full is a no-op). No torch, no checkpoint."""

    def __init__(self, intr, conf=0.9):
        self.intr = intr
        self.conf = conf
        self._pix = None

    def set_pixels(self, pix):
        self._pix = np.asarray(pix, dtype=float)

    def detect(self, rgb, depth=None, intr=None):
        crop = CropTransform(x0=0, y0=0, side=self.intr.width, res=self.intr.width)
        return DetectorResult(xy_det=self._pix.copy(), crop=crop, confidence=self.conf,
                              order=tuple(LUNG_ORDER))


# ── _umeyama: known recovery ───────────────────────────────────────────────────
def test_umeyama_recovers_known_rst_exactly():
    """Clean correspondences dst = s*R*src + t must be inverted to machine precision."""
    rng = np.random.default_rng(11)
    for _ in range(5):
        R0 = _rand_rot(rng)
        s0 = float(rng.uniform(0.5, 2.0))
        t0 = rng.uniform(-1.0, 1.0, size=3)
        src = rng.standard_normal((N, 3))
        dst = s0 * (R0 @ src.T).T + t0
        R, s, t = _umeyama(src, dst, np.ones(N))
        assert np.allclose(R, R0, atol=1e-9)               # exact rotation (no noise present)
        assert abs(s - s0) < 1e-9 * max(1.0, s0)
        assert np.allclose(t, t0, atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9          # proper rotation


def test_umeyama_recovered_pose_reprojects_cleanly():
    """The (R,s,t) it returns must actually map src onto dst (residual ~0)."""
    rng = np.random.default_rng(12)
    R0, s0, t0 = _rand_rot(rng), 1.4, np.array([0.2, -0.3, 0.7])
    src = rng.standard_normal((N, 3)) * 0.05
    dst = s0 * (R0 @ src.T).T + t0
    R, s, t = _umeyama(src, dst, np.ones(N))
    resid = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
    assert resid.max() < 1e-9


# ── _umeyama: reflection / degenerate ──────────────────────────────────────────
def test_umeyama_never_returns_reflection():
    """If dst is a MIRRORED copy of src, Umeyama must still return a proper rotation
    (det(R)=+1), never a reflection (det=-1). This is the S[2,2]=-1 correction."""
    rng = np.random.default_rng(13)
    for axis in range(3):
        src = rng.standard_normal((N, 3))
        M = np.eye(3)
        M[axis, axis] = -1.0                                # pure reflection
        dst = (M @ src.T).T
        R, s, t = _umeyama(src, dst, np.ones(N))
        assert np.linalg.det(R) > 0                         # NEVER a reflection
        assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_umeyama_collinear_and_planar_stay_finite():
    """Rank-deficient src (collinear / planar) is degenerate for full 3D pose but must not
    produce NaN/inf; det(R)=+1 and a finite scale/translation are required."""
    rng = np.random.default_rng(14)
    # collinear src
    base = np.linspace(-1, 1, N)[:, None] * np.array([1.0, 0.0, 0.0])
    dst = base * 1.3 + np.array([0.5, -0.2, 0.1])
    R, s, t = _umeyama(base, dst, np.ones(N))
    assert np.isfinite(R).all() and np.isfinite(s) and np.isfinite(t).all()
    assert np.linalg.det(R) > 0
    # planar src (z=0)
    pl = rng.standard_normal((N, 3)); pl[:, 2] = 0.0
    R2, s2, t2 = _umeyama(pl, pl + 0.3, np.ones(N))
    assert np.isfinite(R2).all() and np.isfinite(s2) and np.isfinite(t2).all()


def test_umeyama_all_coincident_src_no_crash():
    """All src points identical (zero variance) → var_s≈0. Must not crash or return NaN/inf;
    scale must stay finite (the code clamps var_s to 1e-9)."""
    rng = np.random.default_rng(15)
    src = np.tile(np.array([1.0, -2.0, 3.0]), (N, 1))
    dst = rng.standard_normal((N, 3))
    R, s, t = _umeyama(src, dst, np.ones(N))
    assert np.isfinite(R).all() and np.isfinite(s) and np.isfinite(t).all()
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_umeyama_zero_weights_no_crash():
    """All-zero weights → W normalized by max(sum,1e-9). Must stay finite, not divide-by-zero."""
    rng = np.random.default_rng(16)
    src = rng.standard_normal((N, 3))
    dst = rng.standard_normal((N, 3))
    R, s, t = _umeyama(src, dst, np.zeros(N))
    assert np.isfinite(R).all() and np.isfinite(s) and np.isfinite(t).all()


# ── _fit_pose: robustness ──────────────────────────────────────────────────────
def test_fit_pose_rejects_gross_outliers_keeps_inliers():
    """SAFETY: 2 of 10 detector points are wildly wrong (asymmetric, ~0.5 m off). The hard-inlier
    refit must drop them so the 8 good points keep an accurate pose. We assert the recovered pose
    reprojects the INLIER canonical points back onto their clean observations to <2 mm, and the
    two outliers come back measured=False.

    Seed 2 is one of the ~13% of random canonical geometries on which 3-iter IRLS plus a single
    hard-inlier refit did NOT recover (scale collapsed, inliers smeared to ~8 cm); the LTS
    concentration refit fixed it. The scan below covers the other geometries."""
    rng = np.random.default_rng(2)
    canon = rng.standard_normal((N, 3)) * 0.05
    tr = _tracker_with_canonical(canon)
    R0, s0, t0 = np.eye(3), 1.0, np.array([0.0, 0.0, 1.2])
    clean = s0 * (R0 @ canon.T).T + t0                       # exact observations
    obs = clean.copy()
    bad = [1, 7]
    obs[1] = clean[1] + np.array([0.5, -0.4, 0.3])          # asymmetric gross outliers
    obs[7] = clean[7] + np.array([-0.6, 0.5, 0.2])
    R, s, t, measured, fit, _geometry = tr._fit_pose(obs)
    recov = s * (R @ canon.T).T + t
    err = np.linalg.norm(recov - clean, axis=1)
    inl = [k for k in range(N) if k not in bad]
    assert err[inl].max() < 2e-3, f"inliers corrupted by outliers: {err[inl].max()*1000:.1f} mm"
    assert not measured[1] and not measured[7]              # outliers flagged unmeasured
    assert all(measured[k] for k in inl)


def test_fit_pose_asymmetric_outlier_scan_is_clean_or_fails_loud():
    """The face27c finding as a scan, not one seed: over 200 random constellations and poses, 2 of
    10 points get a gross ASYMMETRIC offset (0.1-0.6 m per axis). IRLS + a single hard-inlier refit
    corrupts 15-30% of these (Umeyama scale collapse); the LTS-concentration refit must keep the 8
    inliers within 2 mm, measured, with both outliers unmeasured.

    LTS can still settle in a wrong local minimum from a bad IRLS seed (~0.05% of configs; trial 49
    here, and the face27c June code behaves identically). Such a fit must fail LOUD: below the
    Sim(3) support floor of measured points and never an outlier marked measured. Bound the rate."""
    rng = np.random.default_rng(27)
    loud = 0
    for trial in range(200):
        canon = rng.standard_normal((N, 3)) * 0.05
        tr = _tracker_with_canonical(canon)
        R0, s0 = _rand_rot(rng), float(rng.uniform(0.8, 1.2))
        t0 = rng.uniform(-0.3, 0.3, 3) + np.array([0.0, 0.0, 1.2])
        clean = s0 * (R0 @ canon.T).T + t0
        obs = clean.copy()
        bad = rng.choice(N, 2, replace=False)
        obs[bad] += rng.uniform(0.1, 0.6, (2, 3)) * rng.choice([-1.0, 1.0], (2, 3))
        R, s, t, measured, _fit, _geometry = tr._fit_pose(obs)
        err = np.linalg.norm(s * (R @ canon.T).T + t - clean, axis=1)
        inl = np.setdiff1d(np.arange(N), bad)
        assert not measured[bad].any(), f"trial {trial}: outlier marked measured {measured}"
        if err[inl].max() < 2e-3:
            assert measured[inl].all(), f"trial {trial}: clean inlier unmeasured {measured}"
        else:
            loud += 1
            assert measured.sum() < _MATHEMATICAL_SIM3_SUPPORT, (
                f"trial {trial}: corrupted fit ({err[inl].max()*1000:.1f} mm) claims support")
    assert loud <= 2, f"{loud}/200 corrupted fits (pre-LTS rate is 15-30%)"


def test_underdetermined_fit_coasts_instead_of_publishing_a_wrong_pose():
    """The one wrong-minimum case of the scan above (seed 27, trial 49), replayed through process().
    Its fit marks only 1 point measured; applying it anyway published pose_valid=True with points
    229 mm off (pose_snap_m snaps straight onto it). A fit below the Sim(3) floor must count as a
    miss: hold the last pose, coast every point, and track again on the next clean frame."""
    rng = np.random.default_rng(27)
    for _trial in range(50):                     # same draws as the scan, stop at trial 49
        canon = rng.standard_normal((N, 3)) * 0.05
        R0, s0 = _rand_rot(rng), float(rng.uniform(0.8, 1.2))
        t0 = rng.uniform(-0.3, 0.3, 3) + np.array([0.0, 0.0, 1.2])
        clean = s0 * (R0 @ canon.T).T + t0
        obs = clean.copy()
        bad = rng.choice(N, 2, replace=False)
        obs[bad] += rng.uniform(0.1, 0.6, (2, 3)) * rng.choice([-1.0, 1.0], (2, 3))
    tr = _tracker_with_canonical(canon)
    tr.R, tr.s, tr.t = R0.copy(), s0, t0.copy()
    tr._have_pose = True; tr._initialized = True; tr._tracking_state = "ACTIVE"
    frames = iter([clean, obs, clean])
    tr._detect3d = lambda rgb, depth, robot_mask=None: next(frames)
    img, dep = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)

    def err_mm(out):
        return max(np.linalg.norm(p.xyz_camera - clean[k]) for k, p in enumerate(out.points)) * 1000

    assert tr.process(img, dep, t=0.0).n_measured == N
    out = tr.process(img, dep, t=0.033)
    assert tr._fit_pose(obs)[3].sum() < _MATHEMATICAL_SIM3_SUPPORT    # the case still occurs
    assert out.n_measured == 0 and not out.pose_measured_this_frame
    assert all(p.status == "COASTING" for p in out.points)
    assert err_mm(out) < 1.0, f"held pose moved {err_mm(out):.1f} mm on an underdetermined fit"
    out = tr.process(img, dep, t=0.066)
    assert out.n_measured == N and err_mm(out) < 1.0


def test_fit_pose_returns_none_below_min_valid():
    """Fewer finite correspondences than the Sim(3) support floor → _fit_pose returns None (can't
    fit); at the floor it fits. Geometry quality is gated separately from this count."""
    rng = np.random.default_rng(22)
    canon = rng.standard_normal((N, 3)) * 0.05
    tr = _tracker_with_canonical(canon)
    k = _MATHEMATICAL_SIM3_SUPPORT
    obs = np.full((N, 3), np.nan)
    obs[:k - 1] = canon[:k - 1] + np.array([0.0, 0.0, 1.0])
    assert tr._fit_pose(obs) is None
    obs[:k] = canon[:k] + np.array([0.0, 0.0, 1.0])
    assert tr._fit_pose(obs) is not None


def test_fit_pose_ignores_nan_inf_rows():
    """NaN / inf observation rows must be excluded (treated as not-observed), not poison the fit.
    With 8 clean + 2 NaN/inf, the 8 must recover accurately and the 2 be unmeasured."""
    rng = np.random.default_rng(23)
    canon = rng.standard_normal((N, 3)) * 0.05
    tr = _tracker_with_canonical(canon)
    clean = canon + np.array([0.0, 0.0, 1.0])
    obs = clean.copy()
    obs[2] = np.nan
    obs[6] = np.inf
    res = tr._fit_pose(obs)
    assert res is not None
    R, s, t, measured, fit, _geometry = res
    recov = s * (R @ canon.T).T + t
    inl = [k for k in range(N) if k not in (2, 6)]
    assert np.linalg.norm(recov[inl] - clean[inl], axis=1).max() < 1e-6
    assert not measured[2] and not measured[6]


def test_fit_pose_all_inliers_clean_pose():
    """Sanity: a clean similarity-transformed constellation fits with ~0 residual, all measured."""
    rng = np.random.default_rng(24)
    canon = rng.standard_normal((N, 3)) * 0.05
    tr = _tracker_with_canonical(canon)
    R0, s0, t0 = _rand_rot(rng), 1.2, np.array([0.1, 0.0, 1.3])
    obs = s0 * (R0 @ canon.T).T + t0
    R, s, t, measured, fit, _geometry = tr._fit_pose(obs)
    assert measured.all()
    assert fit < 1e-6
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


# ── _build_canonical: robust template ──────────────────────────────────────────
def test_build_canonical_never_blurs_template_with_midmotion_frame():
    """An init frame captured mid-motion (gross per-point garbage) must never blur the template.
    Every point must be stable across the whole init window, so a window holding that frame is
    refused (fail closed); once the window has moved past it, the template is built and matches the
    true template (up to a global similarity) to sub-mm."""
    rng = np.random.default_rng(31)
    canon_true = rng.standard_normal((N, 3)) * 0.05
    obs_list = []
    for i in range(12):
        R, s, t = _rand_rot(rng), float(rng.uniform(0.9, 1.1)), rng.uniform(-0.3, 0.3, 3)
        frame = s * (R @ canon_true.T).T + t + rng.standard_normal((N, 3)) * 1e-3
        if i == 6:
            frame = frame + rng.standard_normal((N, 3)) * 0.2    # mid-motion garbage frame
        obs_list.append(frame)
    for i in range(12):                                          # the window then moves on
        R, s, t = _rand_rot(rng), float(rng.uniform(0.9, 1.1)), rng.uniform(-0.3, 0.3, 3)
        obs_list.append(s * (R @ canon_true.T).T + t + rng.standard_normal((N, 3)) * 1e-3)
    tr = ModelBasedTracker(detector=None, intrinsics=_intr(), config=ModelConfig(init_frames=12))
    all_points = np.ones(N, dtype=bool)
    tr._init_obs = obs_list[:12]
    assert not tr._build_canonical(all_points)
    assert tr.canonical is None
    tr._init_obs = obs_list[12:]
    assert tr._build_canonical(all_points)
    assert tr.canonical is not None and np.isfinite(tr.canonical).all()
    R, s, t = _umeyama(tr.canonical, canon_true, np.ones(N))   # best similarity to truth
    aligned = s * (R @ tr.canonical.T).T + t
    err = np.linalg.norm(aligned - canon_true, axis=1)
    assert np.median(err) < 2e-3, f"garbage init frame blurred template: {np.median(err)*1000:.2f} mm"


# ── _smooth_pose: EMA stability ─────────────────────────────────────────────────
def test_smooth_pose_stays_valid_similarity_over_many_steps():
    """The pose EMA must remain a valid similarity over a long random walk: R orthonormal with
    det=+1, scale bounded and positive, translation finite — no blow-up / drift to a reflection."""
    rng = np.random.default_rng(41)
    tr = ModelBasedTracker(detector=None, intrinsics=_intr(), config=ModelConfig(pose_ema=0.4))
    for _ in range(1500):
        R = _rand_rot(rng)
        s = float(rng.uniform(0.5, 2.0))
        t = rng.uniform(-1.0, 1.0, 3)
        tr._smooth_pose(R, s, t)
        assert np.isfinite(tr.R).all() and np.isfinite(tr.s) and np.isfinite(tr.t).all()
    assert np.abs(tr.R @ tr.R.T - np.eye(3)).max() < 1e-6      # stays orthonormal
    assert abs(np.linalg.det(tr.R) - 1.0) < 1e-6              # stays a proper rotation
    assert 0.4 < tr.s < 2.1                                    # bounded by the inputs' convex hull


def test_smooth_pose_converges_to_constant_target():
    """Feeding the SAME (R,s,t) repeatedly, the EMA must converge to it (no residual offset)."""
    rng = np.random.default_rng(42)
    tr = ModelBasedTracker(detector=None, intrinsics=_intr(), config=ModelConfig(pose_ema=0.5))
    R0, s0, t0 = _rand_rot(rng), 1.3, np.array([0.2, -0.1, 0.9])
    for _ in range(200):
        tr._smooth_pose(R0, s0, t0)
    assert np.allclose(tr.R, R0, atol=1e-3)
    assert abs(tr.s - s0) < 1e-3
    assert np.allclose(tr.t, t0, atol=1e-3)


# ── full tracker via stub detector ──────────────────────────────────────────────
def test_tracker_init_and_track_with_stub_detector():
    """End-to-end (no torch): a stub detector feeds a projected constellation; the tracker should
    initialize and recover the true 3D points to high precision on clean, static input."""
    rng = np.random.default_rng(51)
    intr = _intr()
    P3 = np.column_stack([rng.uniform(-0.1, 0.1, N), rng.uniform(-0.1, 0.1, N), np.full(N, 1.2)])
    pix = intr.project(P3)
    raw = float(1.2 * 1000.0 / intr.depth_scale)              # metres → raw depth units
    depth = np.full((intr.height, intr.width), raw, np.float32)
    rgb = np.zeros((intr.height, intr.width, 3), np.uint8)
    det = _StubDetector(intr); det.set_pixels(pix)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8, min_conf=0.3))
    out = None
    for i in range(20):
        out = tr.process(rgb, depth, t=i * 0.1)
    assert out.initialized
    assert out.n_measured == N
    err = [np.linalg.norm(out.points[k].xyz_camera - P3[k]) for k in range(N)]
    assert np.median(err) < 1e-3                              # sub-mm on clean static input


def test_tracker_rejects_low_confidence_detection():
    """A detection below min_conf must be ignored (never initialize on it)."""
    rng = np.random.default_rng(52)
    intr = _intr()
    P3 = np.column_stack([rng.uniform(-0.1, 0.1, N), rng.uniform(-0.1, 0.1, N), np.full(N, 1.2)])
    depth = np.full((intr.height, intr.width), 1200.0, np.float32)
    rgb = np.zeros((intr.height, intr.width, 3), np.uint8)
    det = _StubDetector(intr, conf=0.05); det.set_pixels(intr.project(P3))   # below min_conf 0.3
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8, min_conf=0.3))
    out = None
    for _ in range(20):
        out = tr.process(rgb, depth, t=0.0)
    assert out is not None and not out.initialized
