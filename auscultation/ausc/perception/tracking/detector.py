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
                 compute_confidence: bool = True, review_scale: float = 24.0) -> None:
        self.res = res
        self.up_bias = up_bias
        self.compute_confidence = compute_confidence
        self.review_scale = review_scale  # flip% mapped to conf=1-flip/review_scale
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
            pl, uv, hl = self.model.forward_with_landmarks(x, (self.hm_res, self.hm_res))
        xy = self._heatmap_xy(hl.float(), self.res)[0].cpu().numpy()
        return xy, pl[0].argmax(0).cpu().numpy(), uv[0].cpu().numpy()

    def _flip_conf(self, im: np.ndarray, xy0: np.ndarray) -> float:
        xyf, _, _ = self._points_from_im(im[:, ::-1, :].copy())
        swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}
        d = [np.hypot(xy0[k, 0] - (self.res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        flip = float(np.mean(d) / self.res * 100.0)
        return float(np.clip(1.0 - flip / self.review_scale, 0.02, 1.0))

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
    z = depth_m
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
    n[depth_m == 0] = 0
    return n


class NormalsDetector(DenseIUVDetector):
    """Geometry detector backed by ``perc_lmh_v7_adapt`` (surface-NORMAL input, garment-invariant).

    Input is the depth-derived normal map (garment colour/texture is absent by construction), so it
    drops the IUV/semantic channel (``part=uv=None``); the tracker then runs on D+F+G. Same DINOv2+
    LoRA architecture as the RGB detector, so model loading is inherited unchanged.
    """

    def _normal_image(self, rgb_full: np.ndarray, depth, intr):
        import cv2
        from .geometry import back_square_crop, _pad_extract
        _, crop = back_square_crop(rgb_full, depth, intr)
        dc = cv2.resize(_pad_extract(depth, crop.x0, crop.y0, crop.side), (self.res, self.res),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32) / 1000.0
        nrm = np.clip(normals_from_depth_m(dc, *_NORM_K) * 127 + 127, 0, 255).astype(np.uint8)
        return nrm, crop

    def _flip_conf(self, im: np.ndarray, xy0: np.ndarray) -> float:
        flip = im[:, ::-1, :].copy()
        flip[..., 0] = 254 - flip[..., 0]                 # mirror nx with the x-flip
        xyf, _, _ = self._points_from_im(flip)
        xyf = self._structure(xyf)
        swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}
        d = [np.hypot(xy0[k, 0] - (self.res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        return float(np.clip(1.0 - float(np.mean(d)) / self.res * 100.0 / self.review_scale, 0.02, 1.0))

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        if rgb_full is None or depth is None or intr is None:
            return None
        nrm, crop = self._normal_image(rgb_full, depth, intr)
        xy, _, _ = self._points_from_im(nrm)
        xy = self._structure(xy)
        conf = self._flip_conf(nrm, xy) if self.compute_confidence else 0.8
        return DetectorResult(xy_det=xy[:len(LUNG_ORDER)], crop=crop, confidence=conf,
                              part=None, uv=None, order=tuple(LUNG_ORDER))
