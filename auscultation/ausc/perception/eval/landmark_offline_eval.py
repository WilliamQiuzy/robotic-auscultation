"""Offline landmark-refinement evaluation harness (2026-08-07 audit, item #C).

This is the **gate** for the audit refinements (#1 crop-stabilization, #4 flip-TTA,
#5 robust-structure / soft-decode, #3 real-focal normals): none of them may be turned on in
the live detector until they are shown, offline, to help and not regress. It measures the three
quantities the roadmap calls for and the headline tracker metric is blind to:

  1. **Per-point 2D static jitter (px)** — frame-to-frame point wobble with the common (whole-crop)
     translation removed. This is what "reduce 2D jitter" means.
  2. **3D reconstruction stability, decomposed (mm)** — total vs *common-mode* (all 10 points move
     together, the crop-flicker signature that penetrates the constellation fit ~2.2×) vs
     *independent* (per-point). Item #1 attacks the common-mode term specifically.
  3. **Accuracy vs ground truth (PCK@thresh + median px)** — so a "stabilization" that quietly
     drifts the points off the anatomy is caught, not rewarded.

The pure-metric half (everything above the ``# ── detector-driven`` line) is numpy-only and runs
in CI on this laptop. The detector-driven half imports torch/cv2 lazily and runs on a GPU host
with the checkpoint + clip; ``python -m ausc.perception.eval.landmark_offline_eval --help``.

Reuses ``benchmark.pck_ci`` and ``temporal_metrics.common_removed_jitter_mm``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .benchmark import median_ci, pck_ci
from .temporal_metrics import common_removed_jitter_mm

# 10 posterior-lung points, level-major R-before-L (must match detector.LUNG_ORDER).
LUNG_ORDER: Tuple[str, ...] = tuple(f"{s}{i}" for i in range(1, 6) for s in ("R", "L"))


# ── pure metrics (numpy-only, CI-testable) ──────────────────────────────────────────────────

def _seq(a: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != dim:
        raise ValueError(f"{name} must have shape (T, N, {dim}); got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 frames")
    return arr


def per_point_jitter_px(xy_seq: np.ndarray) -> Dict[str, object]:
    """Per-point 2D static jitter [px]: frame-to-frame displacement with the per-frame common
    (median-over-points) translation removed. Returns overall median/p90 and a per-point median."""
    xy = _seq(xy_seq, 2, "xy_seq")
    delta = np.diff(xy, axis=0)                                     # (T-1, N, 2)
    complete = np.all(np.isfinite(delta), axis=2)                   # (T-1, N)
    d = np.where(complete[:, :, None], delta, np.nan)
    common = np.nanmedian(d, axis=1, keepdims=True)                 # whole-crop translation
    mag = np.linalg.norm(d - common, axis=2)                        # (T-1, N) px
    overall = mag[np.isfinite(mag)]
    per_pt = np.nanmedian(np.where(complete, mag, np.nan), axis=0)  # (N,)
    return {
        "median_px": float(np.median(overall)) if overall.size else float("nan"),
        "p90_px": float(np.percentile(overall, 90)) if overall.size else float("nan"),
        "per_point_median_px": {LUNG_ORDER[i]: float(per_pt[i]) for i in range(len(per_pt))},
        "sample_count": int(overall.size),
    }


def decompose_3d_jitter_mm(xyz_seq: np.ndarray) -> Dict[str, float]:
    """Decompose 3D frame-to-frame jitter [mm] into total / common-mode / independent.

    * total       — per-point ||Δp|| (raw wobble).
    * common_mode — magnitude of the per-frame common translation (all 10 points in lock-step);
                    this is the crop-flicker term item #1 targets, and it penetrates the
                    constellation fit ~2.2×, so it dominates 3D quality.
    * independent — per-point wobble after removing the common translation
                    (``common_removed_jitter_mm``).
    """
    xyz = _seq(xyz_seq, 3, "xyz_seq")
    delta = np.diff(xyz, axis=0)                                    # (T-1, N, 3) m
    complete = np.all(np.isfinite(delta), axis=2)
    d = np.where(complete[:, :, None], delta, np.nan)
    total = np.linalg.norm(d, axis=2) * 1000.0
    total = total[np.isfinite(total)]
    common_vec = np.nanmedian(d, axis=1)                           # (T-1, 3)
    common_mag = np.linalg.norm(common_vec, axis=1) * 1000.0
    common_mag = common_mag[np.isfinite(common_mag)]
    try:                                                           # degrade to NaN like the siblings
        indep = common_removed_jitter_mm(xyz)                      # (never abort the whole A/B run)
        indep_med, indep_p90 = indep.median_mm, indep.p90_mm
    except ValueError:
        indep_med = indep_p90 = float("nan")
    return {
        "total_median_mm": float(np.median(total)) if total.size else float("nan"),
        "total_p90_mm": float(np.percentile(total, 90)) if total.size else float("nan"),
        "common_median_mm": float(np.median(common_mag)) if common_mag.size else float("nan"),
        "common_p90_mm": float(np.percentile(common_mag, 90)) if common_mag.size else float("nan"),
        "independent_median_mm": indep_med,
        "independent_p90_mm": indep_p90,
    }


def pck_px(errors_px: Sequence[float], thresh_px: float) -> Dict[str, float]:
    """PCK@thresh_px (% of points within thresh) + median px error, both with bootstrap CIs."""
    err = np.asarray(list(errors_px), dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"pck": float("nan"), "pck_lo": float("nan"), "pck_hi": float("nan"),
                "median_px": float("nan"), "n": 0}
    p, plo, phi = pck_ci(err, thresh_px)
    m, mlo, mhi = median_ci(err)
    return {"pck": p, "pck_lo": plo, "pck_hi": phi,
            "median_px": m, "median_lo": mlo, "median_hi": mhi, "n": int(err.size)}


def evaluate_sequence(xy_seq: np.ndarray, xyz_seq: Optional[np.ndarray] = None,
                      errors_px: Optional[Sequence[float]] = None,
                      thresh_px: float = 40.0) -> Dict[str, object]:
    """Full metric bundle for one variant's run. ``errors_px`` and ``thresh_px`` are in FULL-FRAME
    pixels — a single scale shared across variants, so a variant that shrinks the crop (e.g. the
    percentile bbox) is not scored against a tighter per-variant PCK threshold."""
    out: Dict[str, object] = {"jitter2d": per_point_jitter_px(xy_seq)}
    if xyz_seq is not None:
        out["jitter3d"] = decompose_3d_jitter_mm(xyz_seq)
    if errors_px is not None:
        out["accuracy"] = pck_px(errors_px, thresh_px)
    return out


def _get(d: Dict, *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def compare_markdown(named_results: Dict[str, Dict[str, object]], baseline: str = "baseline") -> str:
    """Side-by-side variant table with Δ vs baseline on the levers that matter. Lower is better on
    every column except PCK."""
    lines = ["| variant | cover % | 2D jitter med px | 3D total mm | 3D common mm | 3D indep mm | "
             "PCK % | median px |",
             "|---|---|---|---|---|---|---|---|"]
    base = named_results.get(baseline, {})
    base_cov = _get(base, "coverage", "detect_rate_pct")

    def _cell(val, ref, better_low=True, pct=False):
        if not np.isfinite(val):
            return "—"
        s = f"{val:.1f}" if not pct else f"{val:.0f}"
        if ref is not None and np.isfinite(ref) and ref != 0 and val != ref:
            dpct = (val - ref) / abs(ref) * 100.0
            good = (dpct < 0) if better_low else (dpct > 0)
            mark = "✓" if good else "✗"
            s += f" ({dpct:+.0f}% {mark})"
        return s

    for name, r in named_results.items():
        is_base = name == baseline
        row = [name + ("" if not is_base else " *(ref)*")]
        cov = _get(r, "coverage", "detect_rate_pct")
        cov_cell = "—" if not np.isfinite(cov) else f"{cov:.0f}"
        if not is_base and np.isfinite(cov) and np.isfinite(base_cov) and cov < base_cov - 0.5:
            cov_cell += " ✗"                                 # dropped frames → comparison unfair
        row.append(cov_cell)
        row.append(_cell(_get(r, "jitter2d", "median_px"),
                         None if is_base else _get(base, "jitter2d", "median_px")))
        row.append(_cell(_get(r, "jitter3d", "total_median_mm"),
                         None if is_base else _get(base, "jitter3d", "total_median_mm")))
        row.append(_cell(_get(r, "jitter3d", "common_median_mm"),
                         None if is_base else _get(base, "jitter3d", "common_median_mm")))
        row.append(_cell(_get(r, "jitter3d", "independent_median_mm"),
                         None if is_base else _get(base, "jitter3d", "independent_median_mm")))
        row.append(_cell(_get(r, "accuracy", "pck"),
                         None if is_base else _get(base, "accuracy", "pck"),
                         better_low=False, pct=True))
        row.append(_cell(_get(r, "accuracy", "median_px"),
                         None if is_base else _get(base, "accuracy", "median_px")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n> Δ vs baseline in parens. ✓ = improvement, ✗ = regression. A variant is a "
                 "**deploy candidate** only if (a) coverage is NOT below baseline (a ✗ in cover % "
                 "voids the comparison — it flattered its jitter/PCK by dropping hard frames), "
                 "(b) PCK does not drop, and (c) at least one jitter/3D column improves. "
                 "Common-mode mm is the item-#1 lever; 2D jitter med px is #4/#5. Errors are "
                 "full-frame px (one scale for all variants).")
    return "\n".join(lines)


# ── detector-driven (GPU host: torch/cv2 lazy) ──────────────────────────────────────────────

def lift_xyz(full_xy: np.ndarray, depth: np.ndarray, intr, win: int = 2) -> np.ndarray:
    """(N,2) full-frame px → (N,3) camera-frame metres via ±win median depth + pinhole unproject
    (mirrors the tracker's lifting). NaN where no valid depth."""
    from ..tracking.geometry import sample_depth_m
    z = sample_depth_m(depth, full_xy, intr.depth_scale, win=win)
    xy = np.atleast_2d(np.asarray(full_xy, dtype=np.float64))
    x = (xy[:, 0] - intr.cx) * z / intr.fx
    y = (xy[:, 1] - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1)


def run_variant(detect_fn: Callable, frames: List[Tuple[np.ndarray, np.ndarray]], intr,
                gt_by_frame: Optional[Dict[int, Dict[str, Sequence[float]]]] = None,
                thresh_px: float = 40.0) -> Dict[str, object]:
    """Drive one detector variant over ``frames`` = list of (rgb, depth). Collects the structured
    2D points (crop→full space), lifts them to 3D, and (if labels given) the per-point FULL-FRAME
    px error. ``detect_fn(rgb, depth, intr) -> DetectorResult`` (or None).

    Also returns a ``coverage`` block (detected-frame count / rate). This matters because the
    jitter/3D/PCK metrics are computed over each variant's *surviving* (detected) frames — a variant
    that quietly drops the hardest (jerkiest / occluded) frames below the conf gate would flatter its
    own jitter and PCK by survivorship. ``compare_markdown`` surfaces coverage and refuses to call
    any variant a deploy candidate if its coverage falls below baseline (the comparison is only fair
    on a shared frame set)."""
    xy_seq, xyz_seq, errs = [], [], []
    n_valid = 0
    for i, (rgb, depth) in enumerate(frames):
        r = detect_fn(rgb, depth, intr)
        if r is None or (r.confidence is not None and r.confidence < 0.05):
            xy_seq.append(np.full((len(LUNG_ORDER), 2), np.nan))
            xyz_seq.append(np.full((len(LUNG_ORDER), 3), np.nan))
            continue
        n_valid += 1
        full = r.crop.det_to_full(r.xy_det)                        # (10,2) full-frame px
        xy_seq.append(full)
        xyz_seq.append(lift_xyz(full, depth, intr))
        if gt_by_frame is not None and i in gt_by_frame:
            g = gt_by_frame[i]
            for k, nm in enumerate(LUNG_ORDER):                    # error in FULL-FRAME px (shared
                if nm in g:                                        # scale → cross-variant comparable)
                    errs.append(float(np.hypot(full[k, 0] - g[nm][0],
                                               full[k, 1] - g[nm][1])))
    out = evaluate_sequence(np.stack(xy_seq), np.stack(xyz_seq),
                            errs if gt_by_frame is not None else None, thresh_px)
    out["coverage"] = {"frames": len(frames), "detected": n_valid,
                       "detect_rate_pct": 100.0 * n_valid / max(len(frames), 1)}
    return out


# ── clip loading + CLI (GPU host) ───────────────────────────────────────────────────────────

# 2026-08-07 linbox A/B (raw detector, static window, no tracker/PCK): crop EMA-only -9% 2D jitter
# (a candidate); percentile bbox +17% (dropped); flip_tta neutral-safe; robust_struct +21..48% and
# soft_decode(temp1.5) +3..7% (dropped as defaults). crop_stab now = EMA-only (crop_pct=None default).
_VARIANTS: Dict[str, dict] = {
    "baseline": {},
    "crop_stab": {"stabilize_crop": True},                       # EMA-only (the -9% candidate)
    "crop_pct": {"stabilize_crop": True, "crop_pct": 2.0},       # +EMA percentile (regressed; kept for A/B)
    "flip_tta": {"flip_tta": True},
    "robust_struct": {"robust_structure": True},
    "soft_decode": {"soft_decode": True, "soft_decode_temp": 1.5},
    "candidates": {"stabilize_crop": True, "flip_tta": True},    # the two that don't regress
}


def _load_frames(clip_dir: str, max_frames: Optional[int]):
    import cv2
    import json
    import os
    with open(os.path.join(clip_dir, "intrinsics.json")) as f:
        d = json.load(f)
    from ..tracking.geometry import Intrinsics
    intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"],
                      d.get("depth_scale", 1.0))
    cap = cv2.VideoCapture(os.path.join(clip_dir, "rgb.mp4"))
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        dp = os.path.join(clip_dir, "depth", f"d{i:04d}.npy")
        if not os.path.exists(dp):
            dp = os.path.join(clip_dir, "depth", f"depth_{i:06d}.npy")
        if not os.path.exists(dp):
            i += 1
            continue
        frames.append((bgr[:, :, ::-1].copy(), np.load(dp)))
        i += 1
    cap.release()
    return frames, intr


def _load_labels(labels_path: Optional[str]):
    if not labels_path:
        return None
    import json
    with open(labels_path) as f:
        raw = json.load(f)
    # accept {frame_index: {name: [x,y]}} or a list of per-frame dicts
    if isinstance(raw, dict) and all(str(k).lstrip("-").isdigit() for k in raw):
        return {int(k): v for k, v in raw.items()}
    return raw


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Offline landmark-refinement A/B (GPU host).")
    ap.add_argument("--ckpt", required=True, help="perc_lmh_v9_occ/best.pt")
    ap.add_argument("--clip", required=True, help="dir with rgb.mp4 + depth/*.npy + intrinsics.json")
    ap.add_argument("--labels", default=None, help="optional {frame:{name:[x,y]}} GT (full-frame px)")
    ap.add_argument("--variants", default="baseline,crop_stab,flip_tta,robust_struct,soft_decode,all_free")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--thresh-px", type=float, default=25.9)
    ap.add_argument("--out", default=None, help="write metrics json here")
    a = ap.parse_args(argv)

    from ..tracking.detector import NormalsDetector
    frames, intr = _load_frames(a.clip, a.max_frames)
    gt = _load_labels(a.labels)
    print(f"[eval] {len(frames)} frames from {a.clip}; labels={'yes' if gt else 'no'}")

    results: Dict[str, Dict[str, object]] = {}
    for name in a.variants.split(","):
        name = name.strip()
        if name not in _VARIANTS:
            print(f"[eval] skip unknown variant {name!r}")
            continue
        det = NormalsDetector(a.ckpt, half=True, **_VARIANTS[name])
        results[name] = run_variant(det.detect, frames, intr, gt, a.thresh_px)
        j2 = _get(results[name], "jitter2d", "median_px")
        print(f"[eval] {name:14s} 2Djit={j2:.2f}px")

    md = compare_markdown(results)
    print("\n" + md + "\n")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"clip": a.clip, "ckpt": a.ckpt, "results": results}, f, indent=2)
        print(f"[eval] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
