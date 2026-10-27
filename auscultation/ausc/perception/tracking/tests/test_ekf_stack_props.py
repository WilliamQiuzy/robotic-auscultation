import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.tracking.ekf import EKFConfig, PointEKF  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────
TRUTH = np.array([0.1, -0.05, 0.6])


def _converge(ekf, truth, n=80, sigma=0.003, seed=0, dt=1.0 / 30):
    """Feed n noisy measurements of a static target; return per-step position errors."""
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(n):
        ekf.predict(dt)
        z = truth + rng.normal(0, sigma, 3)
        ekf.update(z, var=sigma ** 2)
        errs.append(np.linalg.norm(ekf.position - truth))
    return np.asarray(errs)


def _is_pd(P, tol=0.0):
    w = np.linalg.eigvalsh(0.5 * (P + P.T))
    return w.min() > tol


# ── invariants: convergence & boundedness ─────────────────────────────────────
def test_converges_to_static_target():
    ekf = PointEKF(TRUTH + 0.03, EKFConfig())
    errs = _converge(ekf, TRUTH, n=120, sigma=0.003, seed=0)
    # final estimate within a few mm of truth (noise sigma = 3 mm)
    assert errs[-1] < 3e-3
    # late-window error should be no worse than early-window error (convergence, not divergence)
    assert errs[-20:].mean() <= errs[:20].mean()


def test_stays_bounded_over_many_steps():
    """No divergence over a long run; state and covariance remain finite and small."""
    ekf = PointEKF(TRUTH, EKFConfig())
    errs = _converge(ekf, TRUTH, n=1000, sigma=0.003, seed=7)
    assert np.isfinite(ekf.x).all()
    assert np.isfinite(ekf.P).all()
    assert errs.max() < 0.05          # never wanders >5 cm from a 3 mm-noise target
    assert errs[-1] < 5e-3


def test_position_std_decreases_with_observations():
    ekf = PointEKF(TRUTH, EKFConfig())
    s0 = ekf.position_std
    _converge(ekf, TRUTH, n=50, sigma=0.003, seed=1)
    assert ekf.position_std < s0      # uncertainty shrinks once we observe


# ── covariance positive-definiteness ──────────────────────────────────────────
def test_covariance_stays_pd_after_many_updates():
    ekf = PointEKF(TRUTH, EKFConfig())
    rng = np.random.RandomState(2)
    for _ in range(500):
        ekf.predict(1.0 / 30)
        ekf.update(TRUTH + rng.normal(0, 0.003, 3), var=9e-6)
        assert _is_pd(ekf.P), "covariance lost positive-definiteness"


def test_covariance_symmetry_preserved():
    ekf = PointEKF(TRUTH, EKFConfig())
    rng = np.random.RandomState(3)
    worst = 0.0
    for _ in range(300):
        ekf.predict(1.0 / 30)
        ekf.update(TRUTH + rng.normal(0, 0.003, 3), var=9e-6)
        worst = max(worst, np.max(np.abs(ekf.P - ekf.P.T)))
    assert worst < 1e-9               # P must remain (numerically) symmetric


def test_covariance_pd_with_extremely_confident_measurements():
    """Tiny measurement variance stresses the (I-KH)P update toward singularity."""
    ekf = PointEKF(TRUTH, EKFConfig())
    for _ in range(100):
        ekf.predict(1.0 / 30)
        ekf.update(TRUTH, var=1e-12)   # near-perfect measurement
    assert _is_pd(ekf.P), "P degenerated under high-confidence updates"
    assert np.isfinite(ekf.P).all()


# ── outlier robustness (gating) ───────────────────────────────────────────────
def test_single_huge_outlier_is_rejected():
    ekf = PointEKF(TRUTH, EKFConfig())
    _converge(ekf, TRUTH, n=60, sigma=0.003, seed=4)
    before = ekf.position.copy()
    accepted = ekf.update(TRUTH + np.array([0.3, 0.0, 0.0]), var=9e-6)
    assert accepted is False
    assert np.allclose(ekf.position, before)         # state untouched by rejected meas
    assert ekf.n_rejected >= 1


def test_burst_of_outliers_does_not_blow_up_state():
    """With normal re-grounding (a good measurement each step keeps P tight), a parallel
    stream of 1 m-scale garbage must be gated out — the module's central robustness claim."""
    ekf = PointEKF(TRUTH, EKFConfig())
    _converge(ekf, TRUTH, n=40, sigma=0.003, seed=5)
    rng = np.random.RandomState(5)
    for _ in range(60):
        ekf.predict(1.0 / 30)
        ekf.update(TRUTH + rng.normal(0, 0.003, 3), var=9e-6)     # good measurement
        ekf.update(TRUTH + rng.uniform(-1.0, 1.0, 3), var=9e-6)   # garbage, 1 m scale
    assert np.linalg.norm(ekf.position - TRUTH) < 0.02
    assert np.isfinite(ekf.x).all()


# ── adversarial / degenerate inputs ───────────────────────────────────────────
def test_nan_measurement_is_rejected_not_poisoning():
    ekf = PointEKF(TRUTH, EKFConfig())
    _converge(ekf, TRUTH, n=20, sigma=0.003, seed=6)
    x_before = ekf.x.copy()
    accepted = ekf.update(np.array([np.nan, 0.0, 0.0]), var=9e-6)
    assert accepted is False
    assert np.array_equal(ekf.x, x_before)           # not silently poisoned
    assert np.isfinite(ekf.x).all()


def test_inf_measurement_is_rejected():
    ekf = PointEKF(TRUTH, EKFConfig())
    x_before = ekf.x.copy()
    assert ekf.update(np.array([np.inf, 0.0, 0.0]), var=9e-6) is False
    assert np.array_equal(ekf.x, x_before)


def test_nonpositive_or_nonfinite_var_rejected():
    ekf = PointEKF(TRUTH, EKFConfig())
    assert ekf.update(TRUTH, var=0.0) is False
    assert ekf.update(TRUTH, var=-1.0) is False
    assert ekf.update(TRUTH, var=np.nan) is False
    assert ekf.update(TRUTH, var=np.inf) is False


def test_huge_var_measurement_is_near_noop():
    """An uninformative (huge-variance) but in-gate measurement must barely move the state."""
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(1.0 / 30)
    p0 = ekf.position.copy()
    accepted = ekf.update(np.array([5.0, 5.0, 5.0]), var=1e6)
    assert accepted is True
    assert np.linalg.norm(ekf.position - p0) < 1e-4
    assert _is_pd(ekf.P)


def test_predict_nan_dt_does_not_poison_state():
    """predict(NaN) must not turn the state/covariance into NaN.

    `float(max(np.nan, 1e-3))` returns NaN, so the propagated state becomes
    non-finite unless dt is sanitized. A poisoned filter can never recover.
    """
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(np.nan)
    assert np.isfinite(ekf.x).all(), "predict(NaN) poisoned the state"
    assert np.isfinite(ekf.P).all(), "predict(NaN) poisoned the covariance"


def test_predict_inf_dt_does_not_blow_up():
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(np.inf)
    assert np.isfinite(ekf.x).all(), "predict(inf) blew up the state"
    assert np.isfinite(ekf.P).all(), "predict(inf) blew up the covariance"



def test_non_finite_dt_is_counted_not_silently_absorbed():
    """The non-finite-dt guard is a backstop; each use is counted so a corrupt clock stays visible."""
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(1.0 / 30)
    assert ekf.n_invalid_dt == 0
    ekf.predict(np.nan)
    ekf.predict(np.inf)
    assert ekf.n_invalid_dt == 2


def test_predict_huge_finite_dt_is_clamped():
    """A stalled stream or a clock jump gives a huge but finite dt. Extrapolating the velocity over
    the whole gap throws the point metres away, and dt**2 in F P F^T overflows P to inf."""
    cfg = EKFConfig(max_speed=0.5)
    ekf = PointEKF(TRUTH, cfg)
    ekf.x[3:] = np.array([cfg.max_speed, 0.0, 0.0])   # moving at the speed cap
    ekf.predict(1e6)
    assert np.isfinite(ekf.x).all() and _is_pd(ekf.P)
    assert np.linalg.norm(ekf.position - TRUTH) <= cfg.max_speed * 1.0 + 1e-9, "dt not clamped to 1 s"
    ekf = PointEKF(TRUTH, cfg)
    ekf.predict(1e200)
    assert np.isfinite(ekf.x).all(), "predict(1e200) blew up the state"
    assert np.isfinite(ekf.P).all(), "predict(1e200) overflowed the covariance"
    assert ekf.n_invalid_dt == 0                      # finite dt is clamped, not counted as corrupt


def test_predict_negative_dt_treated_sanely():
    """Negative dt is non-physical; it must be clamped (filter must stay finite/PD)."""
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(-5.0)
    assert np.isfinite(ekf.x).all()
    assert _is_pd(ekf.P)
    # position should not have run backwards by metres from a single bad dt
    assert np.linalg.norm(ekf.position - TRUTH) < 1.0


def test_velocity_clamped_to_max_speed():
    cfg = EKFConfig(max_speed=0.5)
    ekf = PointEKF(np.zeros(3), cfg)
    ekf.x[3:] = np.array([10.0, 10.0, 10.0])   # absurd velocity
    ekf.predict(1.0 / 30)
    assert np.linalg.norm(ekf.velocity) <= cfg.max_speed + 1e-9


def test_position_unaffected_by_independent_axes():
    """A measurement displaced only in x must not drag y/z appreciably (isotropic R)."""
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(1.0 / 30)
    y0, z0 = ekf.position[1], ekf.position[2]
    ekf.update(TRUTH + np.array([0.005, 0.0, 0.0]), var=9e-6)  # small in-gate x offset
    assert abs(ekf.position[1] - y0) < 1e-4
    assert abs(ekf.position[2] - z0) < 1e-4


def test_position_property_returns_copy():
    """Readout must not alias internal state (immutability of the filter's view)."""
    ekf = PointEKF(TRUTH, EKFConfig())
    p = ekf.position
    p[0] += 1.0
    assert ekf.x[0] == TRUTH[0]      # mutating the returned array must not affect state


def test_list_measurement_accepted():
    """z given as a plain list (not ndarray) should be coerced and accepted in-gate."""
    ekf = PointEKF(TRUTH, EKFConfig())
    ekf.predict(1.0 / 30)
    accepted = ekf.update([TRUTH[0], TRUTH[1], TRUTH[2]], var=9e-6)
    assert accepted is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))