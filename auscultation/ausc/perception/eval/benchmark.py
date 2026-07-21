"""Generalization benchmark scaffolding — turns per-point error records into the §9.6 medical-grade
report: PCK / median with bootstrap CIs, per-SUBJECT and per-GARMENT breakdowns (the leave-one-out
tables), and the geometry metrics PCK is blind to (L/R parallelism, per-level width consistency).

Single-subject data today makes the CIs wide / breakdowns trivial — by design: this is the harness so
that the moment a subject-disjoint multi-subject test set lands (T6), the gating numbers auto-produce.

Pure numpy → runs in CI. Record = dict(subject, garment, point, err) where err is mm or px.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _boot(vals: np.ndarray, stat, n_boot: int, alpha: float, rng) -> Tuple[float, float, float]:
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    bs = np.array([stat(vals[i]) for i in idx])
    return float(stat(vals)), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def pck_ci(err: np.ndarray, thresh: float, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """PCK@thresh (% within thresh) with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    return _boot(np.asarray(err) < thresh, lambda v: v.mean() * 100.0, n_boot, alpha, rng)


def median_ci(err: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    return _boot(err, np.median, n_boot, alpha, rng)


def breakdown(records: List[Dict], key: str, thresh: float) -> Dict[str, Tuple]:
    """PCK@thresh ± CI for each value of `key` (e.g. 'subject' → per-subject; 'garment' → LOGO)."""
    out = {}
    for v in sorted({r[key] for r in records}):
        e = np.array([r["err"] for r in records if r[key] == v], float)
        out[v] = pck_ci(e, thresh) + (len(e),)
    return out


def width_consistency(points_by_level: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]) -> Dict[int, Tuple[float, float]]:
    """Per-level L/R 3D width (mm) mean+std across frames — PCK is blind to fan-out; this catches it.
    points_by_level[level] = list of (R_xyz, L_xyz) over frames."""
    out = {}
    for lvl, pairs in points_by_level.items():
        w = np.array([np.linalg.norm(np.asarray(R) - np.asarray(L)) for R, L in pairs]) * 1000
        out[lvl] = (float(w.mean()), float(w.std()))
    return out


def report(records: List[Dict], thresh: float, unit: str = "px") -> str:
    """Markdown report: overall PCK/median ± CI + per-subject + per-garment breakdowns."""
    err = np.array([r["err"] for r in records], float)
    n_subj = len({r.get("subject") for r in records}); n_garm = len({r.get("garment") for r in records})
    p, plo, phi = pck_ci(err, thresh); m, mlo, mhi = median_ci(err)
    L = [f"# Generalization benchmark ({len(records)} obs, {n_subj} subjects × {n_garm} garments)",
         "" if n_subj >= 20 else f"> ⚠ n={n_subj} subjects — below the ≥20–30 medical-grade bar; CIs are wide.",
         f"\n**Overall:** PCK@{thresh:.0f}{unit} = {p:.0f}% [{plo:.0f}, {phi:.0f}] · median = {m:.1f} [{mlo:.1f}, {mhi:.1f}] {unit}",
         "\n## Per-subject", "| subject | PCK | 95% CI | n |", "|---|---|---|---|"]
    for k, (pk, lo, hi, n) in breakdown(records, "subject", thresh).items():
        L.append(f"| {k} | {pk:.0f}% | [{lo:.0f},{hi:.0f}] | {n} |")
    L += ["\n## Per-garment (leave-one-garment-out)", "| garment | PCK | 95% CI | n |", "|---|---|---|---|"]
    for k, (pk, lo, hi, n) in breakdown(records, "garment", thresh).items():
        L.append(f"| {k} | {pk:.0f}% | [{lo:.0f},{hi:.0f}] | {n} |")
    return "\n".join(L)
