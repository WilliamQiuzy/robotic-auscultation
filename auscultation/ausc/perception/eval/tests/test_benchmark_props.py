import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.eval.benchmark import (  # noqa: E402
    breakdown,
    median_ci,
    pck_ci,
    report,
    width_consistency,
)

import pytest  # noqa: E402

# Justification for tolerances:
# - PCK is a percentage in [0,100]; exact arithmetic on counts → atol 1e-6 for value identity.
# - Bootstrap CIs are percentiles of resampled means/medians; with a fixed seed they are bit-exact,
#   so determinism is asserted with ==. Bracketing (lo<=point<=hi) is a structural invariant that
#   must hold for the *empirical* mean/PCK because the point estimate is itself an attainable
#   resample value; we assert it exactly (no tolerance needed).


# ── bootstrap CI: bracketing + determinism ──────────────────────────────────
def test_pck_ci_brackets_point_estimate():
    # 70/100 below thresh → PCK = 70%.
    err = np.concatenate([np.zeros(70), np.full(30, 10.0)])
    p, lo, hi = pck_ci(err, 1.0, n_boot=500, seed=3)
    assert abs(p - 70.0) < 1e-6
    assert lo <= p <= hi
    assert 0.0 <= lo and hi <= 100.0


def test_median_ci_brackets_point_estimate():
    rng = np.random.default_rng(11)
    err = rng.normal(5.0, 1.0, size=200)
    m, lo, hi = median_ci(err, n_boot=500, seed=5)
    assert abs(m - np.median(err)) < 1e-9
    assert lo <= m <= hi


def test_pck_ci_deterministic_same_seed():
    err = np.linspace(0.0, 10.0, 50)
    a = pck_ci(err, 5.0, n_boot=300, seed=42)
    b = pck_ci(err, 5.0, n_boot=300, seed=42)
    assert a == b


def test_median_ci_seed_changes_interval():
    rng = np.random.default_rng(0)
    err = rng.normal(0.0, 1.0, size=300)
    a = median_ci(err, n_boot=300, seed=1)
    b = median_ci(err, n_boot=300, seed=2)
    # point estimate identical; CI bounds should generally differ across seeds.
    assert a[0] == b[0]
    assert (a[1], a[2]) != (b[1], b[2])


def test_pck_ci_monotone_in_threshold():
    # Larger threshold ⇒ at least as many points counted ⇒ PCK non-decreasing.
    rng = np.random.default_rng(7)
    err = rng.uniform(0.0, 10.0, size=400)
    p_small, _, _ = pck_ci(err, 2.0, n_boot=200, seed=0)
    p_large, _, _ = pck_ci(err, 8.0, n_boot=200, seed=0)
    assert p_large >= p_small - 1e-9


# ── degenerate sample sizes ─────────────────────────────────────────────────
def test_pck_ci_single_sample_collapses():
    # One sample → every resample is that sample → CI collapses to the point estimate.
    p, lo, hi = pck_ci(np.array([0.5]), 1.0, n_boot=100, seed=0)
    assert p == 100.0 and lo == 100.0 and hi == 100.0


def test_median_ci_single_sample_collapses():
    m, lo, hi = median_ci(np.array([3.0]), n_boot=100, seed=0)
    assert m == 3.0 and lo == 3.0 and hi == 3.0


def test_pck_ci_empty_returns_nan_not_crash():
    p, lo, hi = pck_ci(np.array([]), 1.0, n_boot=100, seed=0)
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)


def test_median_ci_empty_returns_nan_not_crash():
    m, lo, hi = median_ci(np.array([]), n_boot=100, seed=0)
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


# ── adversarial numeric inputs ──────────────────────────────────────────────
def test_pck_ci_handles_inf_errors():
    # An inf error is simply "not within thresh"; should not poison the percentage.
    err = np.array([0.0, 0.0, np.inf, 0.0])  # 3/4 within thresh = 75%
    p, lo, hi = pck_ci(err, 1.0, n_boot=200, seed=0)
    assert abs(p - 75.0) < 1e-6
    assert lo <= p <= hi


def test_median_ci_drops_nan_not_poison():
    # A single NaN err (failed detection / empty frame) must NOT poison the aggregate: it is
    # dropped, and the median equals that of the finite values.
    err = np.array([1.0, 2.0, np.nan, 4.0])
    m, lo, hi = median_ci(err, n_boot=50, seed=0)
    assert np.isfinite(m) and np.isclose(m, np.median([1.0, 2.0, 4.0]))


# ── breakdown / LOSO-LOGO grouping ──────────────────────────────────────────
def _recs(groups, key, rng):
    return [{key: g, "err": float(rng.uniform(0, 10))} for g in groups]


def test_breakdown_partitions_counts_sum_to_total():
    rng = np.random.default_rng(2)
    groups = ["a", "a", "b", "c", "c", "c"]
    recs = _recs(groups, "subject", rng)
    bd = breakdown(recs, "subject", 5.0)
    counts = {k: v[3] for k, v in bd.items()}
    assert counts == {"a": 2, "b": 1, "c": 3}
    assert sum(counts.values()) == len(recs)


def test_breakdown_each_group_held_out_once_no_leakage():
    # Every record must contribute to exactly one group's error array (no double count,
    # no missing). Reconstruct the multiset of errors from the per-group slices.
    rng = np.random.default_rng(4)
    recs = [{"garment": g, "err": float(i)} for i, g in
            enumerate(["x", "y", "x", "z", "y", "z", "x"])]
    bd = breakdown(recs, "garment", 100.0)
    # PCK with huge thresh → all within → 100% in every group regardless of values.
    for k, (pk, lo, hi, n) in bd.items():
        assert pk == 100.0
    # group membership is a true partition of indices
    expected = {"x": 3, "y": 2, "z": 2}
    assert {k: v[3] for k, v in bd.items()} == expected


def test_breakdown_keys_sorted():
    recs = [{"subject": s, "err": 1.0} for s in ["c", "a", "b", "a"]]
    bd = breakdown(recs, "subject", 5.0)
    assert list(bd.keys()) == ["a", "b", "c"]


def test_breakdown_single_group():
    recs = [{"subject": "only", "err": float(e)} for e in [0.0, 0.0, 9.0, 9.0]]
    bd = breakdown(recs, "subject", 5.0)
    assert set(bd.keys()) == {"only"}
    pk, lo, hi, n = bd["only"]
    assert n == 4 and abs(pk - 50.0) < 1e-6 and lo <= pk <= hi


def test_breakdown_skips_records_missing_key():
    # A record lacking the grouping key is skipped (consistent with report()'s r.get header
    # counts) rather than crashing mid-iteration on a partially-labeled record set.
    recs = [{"subject": "s1", "err": 1.0}, {"err": 2.0}]
    out = breakdown(recs, "subject", 5.0)
    assert set(out.keys()) == {"s1"} and out["s1"][-1] == 1


# ── width / fan-out metric ──────────────────────────────────────────────────
def test_width_consistency_flags_fanout_vs_parallel():
    # Parallel column: constant L/R separation → ~0 std.
    parallel = [(np.array([0.05, 0.0, 0.0]), np.array([-0.05, 0.0, 0.0])) for _ in range(6)]
    # Fanned-out column: separation grows frame over frame → large std.
    fan = [(np.array([0.05 + 0.01 * i, 0.0, 0.0]),
            np.array([-0.05 - 0.01 * i, 0.0, 0.0])) for i in range(6)]
    out = width_consistency({1: parallel, 2: fan})
    par_mean, par_std = out[1]
    fan_mean, fan_std = out[2]
    assert par_std < 1e-6
    assert fan_std > par_std
    assert fan_std > 10.0  # mm — clearly fanned out


def test_width_consistency_units_are_mm():
    # 0.1 m separation → 100 mm.
    pairs = [(np.array([0.05, 0.0, 0.0]), np.array([-0.05, 0.0, 0.0]))]
    out = width_consistency({0: pairs})
    mean, std = out[0]
    assert abs(mean - 100.0) < 1e-9


def test_width_consistency_single_frame_zero_std():
    pairs = [(np.array([0.03, 0.04, 0.0]), np.array([0.0, 0.0, 0.0]))]  # norm 0.05 m
    mean, std = width_consistency({0: pairs})[0]
    assert abs(mean - 50.0) < 1e-9 and std == 0.0


# ── report ──────────────────────────────────────────────────────────────────
def test_report_empty_records_does_not_crash():
    # Empty input is a legitimate degenerate case (no observations collected yet).
    md = report([], 5.0, unit="mm")
    assert isinstance(md, str)
    assert "0 obs" in md
    assert "nan" in md.lower()


def test_report_contains_groups_and_overall():
    rng = np.random.default_rng(9)
    recs = []
    for s in ["s1", "s2"]:
        for g in ["shirt", "gown"]:
            for _ in range(5):
                recs.append({"subject": s, "garment": g, "err": float(rng.uniform(0, 8))})
    md = report(recs, 5.0, unit="mm")
    assert "Per-subject" in md and "Per-garment" in md
    assert "s1" in md and "s2" in md
    assert "shirt" in md and "gown" in md
    assert "2 subjects" in md


def test_report_single_sample_single_group():
    recs = [{"subject": "s1", "garment": "g1", "err": 2.0}]
    md = report(recs, 5.0)
    assert "1 obs" in md
    assert "s1" in md and "g1" in md