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
    back_present: Optional[bool] = None  # explicit geometry gate; None if detector cannot decide
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
                 half: bool = False, stabilize_crop: bool = False, crop_pct: Optional[float] = None,
                 flip_tta: bool = False, flip_tta_guard_px: float = 8.0,
                 robust_structure: bool = False, soft_decode: bool = False,
                 soft_decode_temp: float = 1.0, real_focal_normals: bool = False) -> None:
        self.res = res
        self.up_bias = up_bias
        self.compute_confidence = compute_confidence
        self.review_scale = review_scale  # flip% mapped to conf=1-flip/review_scale
        self.half = half                  # fp16 autocast inference (~2x faster on the GPU)
        # ── 2026-08-07 audit refinements — ALL default OFF (live path stays byte-identical) ──
        self.stabilize_crop = stabilize_crop        # #1 temporal deadband-EMA on the crop box
        # crop_pct default None (EMA-only): the 2026-08-07 linbox A/B showed the 2/98 percentile
        # bbox REGRESSES (+17% jitter — it shrinks the crop off the model's training distribution),
        # while EMA-only stabilization HELPS (-9% raw 2D jitter). Percentile is opt-in, off by default.
        self.crop_pct = crop_pct
        self.flip_tta = flip_tta                    # #4 fuse the (already-computed) flipped pass
        self.flip_tta_guard_px = flip_tta_guard_px  #    only average where the two passes agree
        # #3 real per-frame effective-focal normals instead of fixed K=560. RETRAIN-COUPLED:
        # the model was trained on K=560 normals, so turning this ON without a matching retrain
        # (stage1/stage2 with real f_eff) DEGRADES accuracy (train/infer distribution mismatch).
        # Kept default OFF; only enable together with the retrained checkpoint.
        self.real_focal_normals = real_focal_normals
        self._crop_stab = None                      #    lazily-built CropStabilizer (per session)
        self._import_repo()
        import torch  # noqa: F401
        self._torch = __import__("torch")
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.hm_res = res // 4
        from iuv_model import DenseIUVModel, apply_lora  # type: ignore
        _tlh = __import__("train_landmark_human")
        if soft_decode:                             # #5 soft-argmax (DSNT) sub-pixel decode
            self._heatmap_xy = (lambda hl, res, _f=_tlh.heatmap_xy_soft, _t=soft_decode_temp:
                                _f(hl, res, temp=_t))
        else:
            self._heatmap_xy = _tlh.heatmap_xy
        self._structure = _tlh.structure_points_robust if robust_structure else _tlh.structure_points
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
        xyf = self._structure(xyf)
        swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}
        d = [np.hypot(xy0[k, 0] - (self.res - 1 - xyf[swap[k], 0]), xy0[k, 1] - xyf[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        flip = float(np.mean(d) / self.res * 100.0)
        if not np.isfinite(flip):
            return 0.0
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
    valid = np.isfinite(depth_m) & (depth_m > 0)
    z = np.where(valid, depth_m, 0.0).astype(np.float32, copy=False)
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]
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
    n[~valid] = 0
    n[~np.isfinite(n)] = 0
    return n


def _fill_depth_holes(depth_m: np.ndarray) -> np.ndarray:
    """Nearest-valid fill of depth holes (depth<=0) so normals stay continuous. Real Orbbec depth has
    ~25%+ holes (worse with dark clothing / range / lighting); unfilled holes crater the detector
    (PCK@5 100→11 at 40% holes), filling recovers it to ~100% up to 55% holes."""
    m = (depth_m <= 0) | ~np.isfinite(depth_m)
    if not m.any():
        return depth_m
    if m.all():
        return np.zeros_like(depth_m)
    try:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
        return depth_m[tuple(idx)]
    except Exception:  # noqa: BLE001 - fallback: morphological nearest fill
        import cv2
        valid = (~m).astype(np.uint8)
        d = np.where(m, 0, depth_m).copy()
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
        if getattr(self, "stabilize_crop", False):          # #1 percentile bbox + temporal deadband-EMA
            if getattr(self, "_crop_stab", None) is None:
                from .geometry import CropStabilizer
                self._crop_stab = CropStabilizer()
            _, crop = back_square_crop(rgb_full, depth, intr, pct=self.crop_pct,
                                       stabilizer=self._crop_stab)
        else:
            _, crop = back_square_crop(rgb_full, depth, intr)
        dc = cv2.resize(_pad_extract(depth, crop.x0, crop.y0, crop.side), (self.res, self.res),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32)
        dc *= float(intr.depth_scale) / 1000.0
        plausible = (not self.plausibility_gate) or self._back_plausible(dc)   # on RAW depth (pre-fill)
        if getattr(self, "fill_holes", False):
            dc = _fill_depth_holes(dc)    # else holes → gray normal gaps → detector fails (PCK 100→11 at 40% holes)
        if getattr(self, "real_focal_normals", False):    # #3 true crop-space focal (RETRAIN-COUPLED)
            side = max(int(crop.side), 1)
            kk = (float(intr.fx) * self.res / side, float(intr.fy) * self.res / side,
                  self.res / 2.0, self.res / 2.0)
        else:
            kk = _NORM_K                                  # training K=560 (matches current checkpoint)
        nrm = np.clip(normals_from_depth_m(dc, *kk) * 127 + 127, 0, 255).astype(np.uint8)
        return nrm, crop, plausible

    @staticmethod
    def _lr_swap() -> dict:
        return {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
                for nm in LUNG_ORDER}

    def _flip_pass(self, im: np.ndarray, xy0_struct: np.ndarray):
        """Single flipped forward pass. Returns ``(confidence, aligned_raw_flipped_xy)``:
          * ``confidence`` — flip-consistency scalar, computed EXACTLY as the old ``_flip_conf``
            (structured flipped points vs the structured original), so it is unchanged.
          * ``aligned_raw_flipped_xy`` — the (10,2) coords the confidence pass computes and then
            discards: the flipped detections un-mirrored (``res-1-x``) and L/R-swapped back into
            the original frame, pre-structure. Flip-TTA (#4) fuses these back in — free, since this
            pass already runs whenever confidence is on.
        """
        flip = im[:, ::-1, :].copy()
        flip[..., 0] = 254 - flip[..., 0]                 # mirror nx with the x-flip
        xyf, _, _ = self._points_from_im(flip)            # native flipped order, raw
        swap = self._lr_swap()
        xyf_struct = self._structure(xyf)
        d = [np.hypot(xy0_struct[k, 0] - (self.res - 1 - xyf_struct[swap[k], 0]),
                      xy0_struct[k, 1] - xyf_struct[swap[k], 1])
             for k in range(len(LUNG_ORDER))]
        flip_pct = float(np.mean(d)) / self.res * 100.0
        conf = 0.0 if not np.isfinite(flip_pct) else float(
            np.clip(1.0 - flip_pct / self.review_scale, 0.02, 1.0))
        aligned = np.empty((len(LUNG_ORDER), 2), dtype=np.float64)
        for k in range(len(LUNG_ORDER)):                  # raw flipped points re-aligned to xy0 frame
            aligned[k, 0] = self.res - 1 - xyf[swap[k], 0]
            aligned[k, 1] = xyf[swap[k], 1]
        return conf, aligned

    def _fuse_flip_tta(self, xy0: np.ndarray, xyf_aligned: np.ndarray) -> np.ndarray:
        """Per-point average of the two passes where they agree (√2 variance reduction); keep the
        original estimate where they disagree by > ``flip_tta_guard_px`` (occlusion / one bad pass)."""
        out = xy0.copy()
        d = np.hypot(xy0[:, 0] - xyf_aligned[:, 0], xy0[:, 1] - xyf_aligned[:, 1])
        ok = np.isfinite(d) & (d <= self.flip_tta_guard_px)
        out[ok] = 0.5 * (xy0[ok] + xyf_aligned[ok])
        return out

    def _flip_conf(self, im: np.ndarray, xy0: np.ndarray) -> float:
        return self._flip_pass(im, xy0)[0]

    def detect(self, rgb_full: np.ndarray, depth=None, intr=None) -> Optional[DetectorResult]:
        if rgb_full is None or depth is None or intr is None:
            return None
        nrm, crop, plausible = self._normal_image(rgb_full, depth, intr)
        xy_raw, _, _ = self._points_from_im(nrm)
        xy_struct = self._structure(xy_raw)               # original structured estimate (baseline xy)
        conf = 0.8
        xy = xy_struct
        _flip_tta = getattr(self, "flip_tta", False)
        if self.compute_confidence or _flip_tta:          # one flipped pass powers both
            conf_v, xyf_aligned = self._flip_pass(nrm, xy_struct)
            if self.compute_confidence:
                conf = conf_v
            if _flip_tta:                                 # #4 fuse then re-structure (else unchanged)
                xy = self._structure(self._fuse_flip_tta(xy_raw, xyf_aligned))
        if not plausible:
            conf = 0.0            # OOD fail-safe: not a body-plausible back → gate rejects (no action on garbage)
        return DetectorResult(xy_det=xy[:len(LUNG_ORDER)], crop=crop, confidence=conf,
                              back_present=bool(plausible),
                              part=None, uv=None, order=tuple(LUNG_ORDER))
