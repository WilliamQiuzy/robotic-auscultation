"""Landmark detector channel (the **D** + **S** evidence sources).

Wraps the production single-frame model (``perc_lmh_v4``: DINOv2+LoRA backbone, landmark
heatmap head + frozen IUV head) behind a tiny ``LandmarkDetector`` protocol so the tracker can
be unit-tested with a synthetic detector — no torch / GPU / weights needed for the core logic.

A detection yields, for the 10 posterior-lung points:
  - ``xy_det``   structured 2D points in detector space (res×res, after ``center_square``)
  - ``part`` / ``uv``  the IUV maps (for the semantic anchor channel)
  - ``crop``     the affine back to full-frame pixels (for depth lookup)
  - ``confidence`` from flip-consistency (lower flip disagreement ⇒ higher confidence)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .geometry import CropTransform

# 10 posterior-lung points, interleaved by level — must match the model checkpoint "order".
LUNG_ORDER: List[str] = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]
LUNG_INDEX = {nm: i for i, nm in enumerate(LUNG_ORDER)}


@dataclass(frozen=True)
class DetectorResult:
    """Output of one detector forward pass (detector space unless noted)."""

    xy_det: np.ndarray              # (10,2) structured points in res×res space
    crop: CropTransform            # det space ↔ full-frame pixels
    confidence: float              # 0..1 (from flip-consistency); 1 = very confident
    part: Optional[np.ndarray] = None   # (res,res) int part id (0=bg, 1=back torso ...)
    uv: Optional[np.ndarray] = None     # (2,res,res) U,V in [0,1]
    order: Tuple[str, ...] = tuple(LUNG_ORDER)


class LandmarkDetector(Protocol):
    """Minimal contract the tracker depends on (so it can be mocked in tests)."""

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        ...


def iuv_locate(part: np.ndarray, uv: np.ndarray, addr: Tuple[int, float, float],
               tol: float = 0.06) -> Optional[Tuple[float, float]]:
    """Find the detector-space pixel whose IUV (part,U,V) best matches ``addr``.

    Returns (x, y) or None if no pixel of the target part is within ``tol`` in (U,V).
    This is the **semantic** re-grounding: it locates a point by its body-surface address,
    which is invariant to camera/patient pose.
    """
    p_id, u_t, v_t = addr
    mask = part == int(p_id)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    du = uv[0][ys, xs] - u_t
    dv = uv[1][ys, xs] - v_t
    d2 = du * du + dv * dv
    j = int(np.argmin(d2))
    if d2[j] > tol * tol:
        return None
    return float(xs[j]), float(ys[j])


class DenseIUVDetector:
    """Production detector backed by ``perc_lmh_v4``. torch is imported lazily."""

    def __init__(self, ckpt: str, encoder: str = "dinov2_vitb14", res: int = 518,
                 up_bias: float = 0.12, device: Optional[str] = None,
                 compute_confidence: bool = True, review_scale: float = 24.0,
                 half: bool = False) -> None:
        self.res = res
        self.up_bias = up_bias
        self.compute_confidence = compute_confidence
        self.review_scale = review_scale  # flip% mapped to conf=1-flip/review_scale
        self.half = half                  # fp16 autocast inference (~2x faster on the GPU)
        self._import_repo()
        import torch  # noqa: F401
        self._torch = __import__("torch")
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.hm_res = res // 4
        from iuv_model import DenseIUVModel, apply_lora  # type: ignore
        self._heatmap_xy = __import__("train_landmark_human").heatmap_xy
        self._structure = __import__("train_landmark_human").structure_points
        from dataset import IMAGENET_MEAN, IMAGENET_STD  # type: ignore
        self._mean, self._std = IMAGENET_MEAN, IMAGENET_STD
        ck = self._torch.load(ckpt, map_location=self.device)
        order = ck.get("order", LUNG_ORDER)[:len(LUNG_ORDER)]
        if list(order) != LUNG_ORDER:
            raise ValueError(f"checkpoint order {order} != expected {LUNG_ORDER}")
        m = DenseIUVModel(encoder=encoder, frozen_encoder=False, n_landmarks=len(LUNG_ORDER))
        apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"])
        m.load_state_dict(ck["lora_head"], strict=False)
        self.model = m.to(self.device).eval()

    @staticmethod
    def _import_repo() -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # auscultation/
        for p in ("ausc", "ausc/models", "ausc/training", "scripts"):
            d = os.path.join(root, p)
            if d not in sys.path:
                sys.path.insert(0, d)

    def _preprocess(self, rgb_full: np.ndarray, depth=None, intr=None
                    ) -> Tuple["np.ndarray", CropTransform]:
        import cv2
        if depth is not None and intr is not None:        # depth-centered, pad-not-clamp, invertible
            from .geometry import back_square_crop
            return back_square_crop(rgb_full, depth, intr)
        h, w = rgb_full.shape[:2]                          # RGB-only fallback (no depth)
        crop = CropTransform.center_square(h, w, self.res, self.up_bias)
        sq = rgb_full[crop.y0:crop.y0 + crop.side, crop.x0:crop.x0 + crop.side]
        im = cv2.resize(sq, (self.res, self.res), interpolation=cv2.INTER_AREA)
        return im, crop

    def _points_from_im(self, im: np.ndarray):
        t = self._torch
        x = t.from_numpy(((im / 255.0 - self._mean) / self._std)
                         .transpose(2, 0, 1)[None].astype(np.float32)).to(self.device)
        with t.no_grad():
            if self.half and self.device == "cuda":
                with t.autocast("cuda", dtype=t.float16):
                    pl, uv, hl = self.model.forward_with_landmarks(x, (self.hm_res, self.hm_res))
            else:
                pl, uv, hl = self.model.forward_with_landmarks(x, (self.hm_res, self.hm_res))
        xy = self._heatmap_xy(hl.float(), self.res)[0].cpu().numpy()
        return xy, pl[0].argmax(0).cpu().numpy(), uv[0].cpu().numpy()

    def _flip_conf(self, im: np.ndarray, xy0: np.ndarray) -> float:
        xyf, _, _ = self._points_from_im(im[:, ::-1, :].copy())
        xyf = self._structure(xyf)            # structure flipped pts too (symmetric with the forward pass)
        swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}
        d = [np.hypot(xy0[k, 0] - (self.res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        c = 1.0 - float(np.mean(d)) / self.res * 100.0 / self.review_scale
        return float(np.clip(c, 0.02, 1.0)) if np.isfinite(c) else 0.0   # NaN flip pass ⇒ fail closed

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        if rgb_full is None:
            return None
        im, crop = self._preprocess(rgb_full, depth, intr)
        xy, part, uv = self._points_from_im(im)
        xy = self._structure(xy)  # enforce R/L parallel + even spacing
        conf = self._flip_conf(im, xy) if self.compute_confidence else 0.8
        return DetectorResult(xy_det=xy[:len(LUNG_ORDER)], crop=crop, confidence=conf,
                              part=part, uv=uv, order=tuple(LUNG_ORDER))


# K of the synthetic normal-map training (518² crop, f=560) — the geometry detector must build
# its input normals with THIS K, not the camera's, to match how the model was trained.
_NORM_K = (560.0, 560.0, 259.0, 259.0)


def normals_from_depth_m(depth_m: np.ndarray, fx: float = 560.0, fy: float = 560.0,
                         cx: float = 259.0, cy: float = 259.0) -> np.ndarray:
    """Per-pixel unit surface normals (camera-facing) from a metric depth map, via neighbour
    cross-products. Output channels are nx,ny,nz; holes (depth==0) → 0."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = np.where(np.isfinite(depth_m), depth_m, 0.0)   # NaN/inf depth → 0 (a hole); never propagate
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    P = np.dstack([x, y, z])
    n = np.zeros_like(P)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    cr = np.cross(dx, dy)
    cr /= np.clip(np.linalg.norm(cr, axis=-1, keepdims=True), 1e-9, None)
    cr[cr[..., 2] > 0] *= -1
    n[1:-1, 1:-1] = cr
    n[z == 0] = 0                                       # holes (incl. former non-finite) → zero normal
    return n


def _fill_depth_holes(depth_m: np.ndarray) -> np.ndarray:
    """Nearest-valid fill of depth holes (depth<=0) so normals stay continuous. Real Orbbec depth has
    ~25%+ holes (worse with dark clothing / range / lighting); unfilled holes crater the detector
    (PCK@5 100→11 at 40% holes), filling recovers it to ~100% up to 55% holes."""
    m = (depth_m <= 0) | ~np.isfinite(depth_m)   # NaN/inf are holes too (NaN <= 0 is False!)
    if not m.any():
        return depth_m
    if m.all():
        return np.zeros_like(depth_m)            # nothing valid → finite zeros, never propagate NaN
    try:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
        return depth_m[tuple(idx)]
    except Exception:  # noqa: BLE001 - fallback: morphological nearest fill
        import cv2
        valid = (~m).astype(np.uint8)
        d = depth_m.copy()
        for _ in range(8):
            if not (d <= 0).any():
                break
            dil = cv2.dilate(d, np.ones((3, 3), np.float32))
            d = np.where(d <= 0, dil, d)
        return d


class NormalsDetector(DenseIUVDetector):
    """Geometry detector backed by ``perc_lmh_v7_adapt`` (surface-NORMAL input, garment-invariant).

    Input is the depth-derived normal map (garment colour/texture is absent by construction), so it
    drops the IUV/semantic channel (``part=uv=None``); the tracker then runs on D+F+G. Same DINOv2+
    LoRA architecture as the RGB detector, so model loading is inherited unchanged.

    Defaults to fp16 inference: measured p50 47 ms (~21 Hz) at 1080p (perception_metrics.json,
    perc_lmh_v9_occ) with IDENTICAL accuracy (held-out PCK@5 100, median 9.0 px unchanged); clears
    the 20 Hz control rate. Pass ``half=False`` to force fp32.
    """

    def __init__(self, *args, half: bool = True, fill_holes: bool = True,
                 plausibility_gate: bool = True, **kwargs):
        super().__init__(*args, half=half, **kwargs)
        self.fill_holes = fill_holes      # fill depth holes before normals (Orbbec ~25%+ holes)
        self.plausibility_gate = plausibility_gate   # OOD fail-safe: conf=0 if the crop isn't a body-back

    def _back_plausible(self, dc_m: np.ndarray) -> bool:
        """Is the crop a plausible patient back (vs empty wall / flat surface / noise)? Hole-robust:
        body has a body-distance median AND a real depth-extent; flat/empty have ~0 range, far wall a
        far median. (Measured: back range ~620mm/median 1.15m; flat/empty 0mm; far wall 2.5m.)"""
        v = dc_m[(dc_m > 0.2) & (dc_m < 3.0)]
        if v.size < 0.12 * dc_m.size:
            return False
        med = float(np.median(v))
        rng = float(np.percentile(v, 90) - np.percentile(v, 10)) * 1000.0
        return (0.4 < med < 1.6) and (rng > 80.0)

    def _normal_image(self, rgb_full: np.ndarray, depth, intr):
        import cv2
        from .geometry import back_square_crop, _pad_extract
        _, crop = back_square_crop(rgb_full, depth, intr)
        # raw depth → metres using the camera's depth_scale (NOT a hardcoded /1000): this matches
        # sample_depth_m / pointmap_from_depth, so both the normal-map input AND the metric
        # _back_plausible OOD gate are correct on any Orbbec profile (depth_scale != 1).
        dc = cv2.resize(_pad_extract(depth, crop.x0, crop.y0, crop.side), (self.res, self.res),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32) * (intr.depth_scale / 1000.0)
        plausible = (not self.plausibility_gate) or self._back_plausible(dc)   # on RAW depth (pre-fill)
        if getattr(self, "fill_holes", False):
            dc = _fill_depth_holes(dc)    # else holes → gray normal gaps → detector fails (PCK 100→11 at 40% holes)
        nrm = np.clip(normals_from_depth_m(dc, *_NORM_K) * 127 + 127, 0, 255).astype(np.uint8)
        return nrm, crop, plausible

    def _flip_conf(self, im: np.ndarray, xy0: np.ndarray) -> float:
        flip = im[:, ::-1, :].copy()
        flip[..., 0] = 254 - flip[..., 0]                 # mirror nx with the x-flip
        xyf, _, _ = self._points_from_im(flip)
        xyf = self._structure(xyf)
        swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}
        d = [np.hypot(xy0[k, 0] - (self.res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        c = 1.0 - float(np.mean(d)) / self.res * 100.0 / self.review_scale
        return float(np.clip(c, 0.02, 1.0)) if np.isfinite(c) else 0.0   # NaN flip pass ⇒ fail closed

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        if rgb_full is None or depth is None or intr is None:
            return None
        nrm, crop, plausible = self._normal_image(rgb_full, depth, intr)
        xy, _, _ = self._points_from_im(nrm)
        xy = self._structure(xy)
        conf = self._flip_conf(nrm, xy) if self.compute_confidence else 0.8
        if not plausible:
            conf = 0.0            # OOD fail-safe: not a body-plausible back → gate rejects (no action on garbage)
        return DetectorResult(xy_det=xy[:len(LUNG_ORDER)], crop=crop, confidence=conf,
                              part=None, uv=None, order=tuple(LUNG_ORDER))
