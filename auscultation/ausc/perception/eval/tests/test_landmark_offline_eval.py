"""Offline landmark-refinement harness metrics (2026-08-07 audit item #C).

Pins the three levers the harness must measure correctly, because the whole deploy gate rests on
them: 2D per-point jitter, the total/common/independent 3D decomposition, and PCK.
"""
import numpy as np

from ausc.perception.eval.landmark_offline_eval import (
    LUNG_ORDER,
    compare_markdown,
    decompose_3d_jitter_mm,
    evaluate_sequence,
    pck_px,
    per_point_jitter_px,
    run_variant,
)


def _rng():
    return np.random.RandomState(0)


# ── 2D jitter ────────────────────────────────────────────────────────────────
def test_2d_jitter_ignores_pure_common_translation():
    # 10 points that translate together every frame (crop drift) but never wobble individually
    base = _rng().uniform(50, 450, size=(10, 2))
    seq = np.stack([base + np.array([3.0 * t, -2.0 * t]) for t in range(20)])
    m = per_point_jitter_px(seq)
    assert m["median_px"] < 1e-6                       # common-mode removed → ~zero independent jitter


def test_2d_jitter_detects_per_point_noise():
    rng = _rng()
    base = rng.uniform(50, 450, size=(10, 2))
    seq = np.stack([base + rng.normal(0, 2.0, size=(10, 2)) for _ in range(40)])
    m = per_point_jitter_px(seq)
    assert m["median_px"] > 1.0
    assert set(m["per_point_median_px"]) == set(LUNG_ORDER)


# ── 3D decomposition ─────────────────────────────────────────────────────────
def test_3d_decomposition_isolates_common_vs_independent():
    rng = _rng()
    base = rng.uniform(-0.1, 0.1, size=(10, 3)) + np.array([0, 0, 1.0])
    # pure common-mode: whole constellation shifts identically each frame
    common = np.stack([base + rng.normal(0, 0.005, size=(1, 3)) for _ in range(30)])
    dc = decompose_3d_jitter_mm(common)
    assert dc["common_median_mm"] > dc["independent_median_mm"] * 3      # common dominates
    # pure independent: each point wobbles on its own, no shared motion
    indep = np.stack([base + rng.normal(0, 0.005, size=(10, 3)) for _ in range(30)])
    di = decompose_3d_jitter_mm(indep)
    assert di["independent_median_mm"] > di["common_median_mm"] * 2      # independent dominates


# ── PCK ──────────────────────────────────────────────────────────────────────
def test_pck_px_counts_within_threshold():
    err = [1.0, 2.0, 3.0, 100.0]        # 3/4 within 25.9
    r = pck_px(err, 25.9)
    assert abs(r["pck"] - 75.0) < 1e-6
    assert r["n"] == 4


# ── comparison table ─────────────────────────────────────────────────────────
def test_compare_markdown_marks_improvement_and_regression():
    good = {"jitter2d": {"median_px": 1.0}, "jitter3d": {"total_median_mm": 1.0,
            "common_median_mm": 0.2, "independent_median_mm": 0.8},
            "accuracy": {"pck": 98.0, "median_px": 8.0},
            "coverage": {"detect_rate_pct": 100.0}}
    base = {"jitter2d": {"median_px": 2.0}, "jitter3d": {"total_median_mm": 2.0,
            "common_median_mm": 0.7, "independent_median_mm": 1.0},
            "accuracy": {"pck": 98.0, "median_px": 9.0},
            "coverage": {"detect_rate_pct": 100.0}}
    md = compare_markdown({"baseline": base, "crop_stab": good})
    assert "crop_stab" in md and "baseline" in md
    assert "✓" in md                                   # crop_stab improves jitter → a tick appears


def test_compare_markdown_flags_coverage_drop_as_survivorship():
    # a variant that "wins" every jitter column but detected fewer frames must be flagged (✗ in
    # coverage) so the gate does not reward survivorship bias.
    base = {"jitter2d": {"median_px": 2.0}, "jitter3d": {"total_median_mm": 2.0,
            "common_median_mm": 0.7, "independent_median_mm": 1.0},
            "accuracy": {"pck": 90.0, "median_px": 9.0},
            "coverage": {"detect_rate_pct": 100.0}}
    dropper = {"jitter2d": {"median_px": 0.5}, "jitter3d": {"total_median_mm": 0.5,
               "common_median_mm": 0.1, "independent_median_mm": 0.4},
               "accuracy": {"pck": 99.0, "median_px": 3.0},
               "coverage": {"detect_rate_pct": 70.0}}       # dropped 30% of (the hard) frames
    md = compare_markdown({"baseline": base, "dropper": dropper})
    dropper_row = [ln for ln in md.splitlines() if ln.startswith("| dropper")][0]
    assert "✗" in dropper_row                               # coverage regression is surfaced


# ── end-to-end over a mock detector ──────────────────────────────────────────
class _Crop:
    side = 400

    def det_to_full(self, xy):
        return np.asarray(xy, float)                    # identity det→full for the test


class _Res:
    def __init__(self, xy):
        self.xy_det = xy
        self.crop = _Crop()
        self.confidence = 0.9


def test_run_variant_scores_a_steadier_detector_better():
    rng = np.random.RandomState(1)
    gt = rng.uniform(100, 300, size=(10, 2))

    class Intr:
        fx = fy = 500.0
        cx = cy = 256.0
        depth_scale = 1.0

    depth = np.full((512, 512), 1000.0, dtype=np.float32)   # flat 1 m → valid lift everywhere
    frames = [(np.zeros((512, 512, 3), np.uint8), depth) for _ in range(20)]

    def noisy(sd):
        def fn(rgb, dep, intr):
            return _Res(gt + rng.normal(0, sd, size=(10, 2)))
        return fn

    steady = run_variant(noisy(0.5), frames, Intr(), thresh_px=25.9)
    shaky = run_variant(noisy(4.0), frames, Intr(), thresh_px=25.9)
    assert steady["jitter2d"]["median_px"] < shaky["jitter2d"]["median_px"]
    assert "jitter3d" in steady and np.isfinite(steady["jitter3d"]["total_median_mm"])
