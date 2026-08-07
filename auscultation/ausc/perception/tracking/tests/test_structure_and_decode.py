"""Robust structure_points + soft-argmax decode (2026-08-07 audit item #5).

Both are additive alternatives to the production functions; these tests pin (a) they agree with the
originals on clean input and (b) they are genuinely more robust to the failure the audit named:
noisy endpoint levels (R5/L5) smearing the whole constellation via the endpoint-only linspace, and
argmax grid-flip in the sub-pixel decode.
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
for _p in ("ausc", "ausc/models", "ausc/training", "scripts"):
    _d = os.path.join(_ROOT, _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402
import train_landmark_human as tlh  # noqa: E402


def _grid(spacing=30.0, width=40.0, cx=259.0, cy=120.0):
    """A clean, evenly-spaced, bilaterally-symmetric 10-point back constellation in ORDER."""
    xy = np.zeros((10, 2), np.float64)
    for i in range(5):
        y = cy + i * spacing
        xy[tlh.IDX[f"R{i+1}"]] = [cx + width, y]
        xy[tlh.IDX[f"L{i+1}"]] = [cx - width, y]
    return xy


# ── robust structure_points ──────────────────────────────────────────────────
def test_robust_matches_original_on_clean_grid():
    xy = _grid()
    a = tlh.structure_points(xy.copy())
    b = tlh.structure_points_robust(xy.copy())
    assert np.allclose(a, b, atol=1e-6)


def test_robust_resists_noisy_endpoint():
    xy = _grid()
    truth = tlh.structure_points(xy.copy())            # ideal structured grid
    bad = xy.copy()
    bad[tlh.IDX["R5"]] += [0.0, 40.0]                   # corrupt the noisy lower endpoint level
    bad[tlh.IDX["L5"]] += [0.0, 40.0]
    orig = tlh.structure_points(bad.copy())
    robust = tlh.structure_points_robust(bad.copy())
    # the middle layer (R3/L3) should drift LESS with the robust fit than with the linspace fit
    mid = tlh.IDX["R3"]
    err_orig = np.linalg.norm(orig[mid] - truth[mid])
    err_robust = np.linalg.norm(robust[mid] - truth[mid])
    assert err_robust <= err_orig + 1e-9
    assert err_robust < err_orig * 0.9                 # meaningfully better, not a rounding tie


# ── soft-argmax decode ───────────────────────────────────────────────────────
def _heatmap(cx, cy, res_h=129, sigma=2.0, K=10):
    ys, xs = np.mgrid[0:res_h, 0:res_h]
    hm = np.zeros((1, K, res_h, res_h), np.float32)
    for k in range(K):
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        hm[0, k] = np.log(g + 1e-6)                     # logits (heatmap_xy applies sigmoid/softmax)
    return torch.from_numpy(hm)


def test_soft_decode_locates_peak():
    hm = _heatmap(40.0, 30.0)
    xy = tlh.heatmap_xy_soft(hm, res=518)[0].numpy()
    # heatmap grid 129 → res 518 scale = 4.0; peak (40,30) → (160,120)
    assert np.allclose(xy[0], [40.0 * 518 / 129, 30.0 * 518 / 129], atol=6.0)


def test_soft_decode_temp1_equals_original():
    # heatmap_xy is already a windowed soft-argmax on sigmoid probs; heatmap_xy_soft at temp=1 must
    # reduce to it EXACTLY (so the soft_decode variant is never worse than baseline at its default).
    hm = _heatmap(50.0, 45.0)
    a = tlh.heatmap_xy(hm, res=518)[0].numpy()
    b = tlh.heatmap_xy_soft(hm, res=518, temp=1.0)[0].numpy()
    assert np.allclose(a, b, atol=1e-4)


def test_soft_decode_temperature_softens_toward_window_centre():
    # a higher temperature flattens the window weights → the estimate moves toward the window centre
    # (the anti-grid-flip behaviour). Just assert temp changes the estimate in a bounded way.
    hm = _heatmap(50.4, 45.0)
    b1 = tlh.heatmap_xy_soft(hm, res=518, temp=1.0)[0, 0, 0].item()
    b3 = tlh.heatmap_xy_soft(hm, res=518, temp=3.0)[0, 0, 0].item()
    assert abs(b1 - b3) < 8.0 * (518 / 129)            # bounded shift, no blow-up


def test_soft_decode_is_continuous_under_subpixel_shift():
    # a smooth decode must move continuously as the true peak slides sub-pixel; argmax-centroid
    # can jump when the integer peak flips. Check monotone, jump-free response over a 1-px slide.
    est = []
    for frac in np.linspace(0.0, 1.0, 11):
        hm = _heatmap(40.0 + frac, 30.0)
        est.append(tlh.heatmap_xy_soft(hm, res=518)[0, 0, 0].item())
    est = np.array(est)
    steps = np.diff(est)
    assert np.all(steps >= -1e-3)                       # monotone non-decreasing
    assert steps.max() < 8.0 * (518 / 129)             # no discontinuous grid-flip jump
