"""Tests for the generalization benchmark scaffolding (bootstrap CI + breakdowns + width metric)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.eval.benchmark import (breakdown, median_ci, pck_ci,  # noqa: E402
                                            report, width_consistency)


def test_pck_ci_bounds_and_value():
    err = np.concatenate([np.zeros(90), np.full(10, 100.0)])   # 90% within thresh 50
    p, lo, hi = pck_ci(err, 50.0)
    assert abs(p - 90) < 1 and lo <= p <= hi and 0 <= lo and hi <= 100


def test_median_ci_brackets_truth():
    rng = np.random.default_rng(1)
    err = rng.normal(10, 2, 500)
    m, lo, hi = median_ci(err)
    assert lo < 10 < hi and abs(m - 10) < 1


def test_breakdown_per_group():
    recs = ([{"subject": "s1", "garment": "dark", "err": 1.0}] * 10
            + [{"subject": "s2", "garment": "yellow", "err": 100.0}] * 10)
    bs = breakdown(recs, "subject", 50.0)
    assert bs["s1"][0] == 100.0 and bs["s2"][0] == 0.0          # s1 all-pass, s2 all-fail
    bg = breakdown(recs, "garment", 50.0)
    assert set(bg) == {"dark", "yellow"}


def test_width_consistency_flags_fanout():
    stable = {5: [(np.array([0.1, 0, 1.0]), np.array([-0.1, 0, 1.0]))] * 5}      # constant 200mm
    fanned = {5: [(np.array([0.1 + 0.05 * i, 0, 1.0]), np.array([-0.1, 0, 1.0])) for i in range(5)]}
    assert width_consistency(stable)[5][1] < 1.0                 # ~0 std
    assert width_consistency(fanned)[5][1] > 10.0                # high std = fan-out


def test_report_warns_below_medical_bar():
    recs = [{"subject": "s1", "garment": "dark", "err": 5.0}] * 20
    md = report(recs, 25.9, unit="px")
    assert "below the" in md and "Per-subject" in md and "Per-garment" in md
