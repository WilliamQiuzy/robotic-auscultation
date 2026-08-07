"""Detector flip-TTA fusion + byte-identity guard (2026-08-07 audit item #4).

The one non-negotiable: with ``flip_tta=False`` (the default) the detector must be byte-identical
to the pre-audit path — same structured xy, same flip-consistency confidence. Flip-TTA, when on,
must (a) reuse the single flipped pass, (b) average the two estimates where they agree, and
(c) fall back to the original where they disagree beyond the guard (occlusion / a bad pass).

Built without loading a checkpoint: we inject ``_normal_image`` / ``_points_from_im`` and drive the
real ``detect`` / ``_flip_pass`` / ``_fuse_flip_tta`` / ``_lr_swap`` methods.
"""
import numpy as np

from ausc.perception.tracking.detector import (
    LUNG_INDEX,
    LUNG_ORDER,
    NormalsDetector,
)


def _make(flip_tta: bool, orig_xy, flip_xy):
    """A NormalsDetector with no model: identity structure, a 2-call points queue (orig, flip)."""
    det = object.__new__(NormalsDetector)
    det.res = 518
    det.compute_confidence = True
    det.review_scale = 24.0
    det.flip_tta = flip_tta
    det.flip_tta_guard_px = 8.0
    det.stabilize_crop = False
    det._structure = lambda a: a                        # identity → reason about exact coords
    queue = {"i": 0, "pts": [np.asarray(orig_xy, float), np.asarray(flip_xy, float)]}

    def _points(im):
        p = queue["pts"][queue["i"]]
        queue["i"] += 1
        return p.copy(), None, None

    det._points_from_im = _points

    class _Crop:
        side = 400
    det._normal_image = lambda a, b, c: (np.zeros((518, 518, 3), np.uint8), _Crop(), True)
    return det


def _flip_of(orig):
    """Build a flipped-frame detection that aligns EXACTLY back onto ``orig`` (perfect agreement).

    align: aligned[k] = (res-1 - flip[swap[k],0], flip[swap[k],1]); we want aligned == orig, so
    flip[swap[k]] = (res-1-orig[k,0], orig[k,1]).
    """
    swap = {LUNG_INDEX[nm]: LUNG_INDEX[("L" + nm[1:] if nm[0] == "R" else "R" + nm[1:])]
            for nm in LUNG_ORDER}
    flip = np.zeros_like(orig)
    for k in range(len(LUNG_ORDER)):
        flip[swap[k]] = [517 - orig[k, 0], orig[k, 1]]
    return flip


def _base_orig():
    rng = np.random.RandomState(0)
    return rng.uniform(120, 400, size=(10, 2))


def test_flip_tta_off_is_byte_identical():
    orig = _base_orig()
    flip = _flip_of(orig)
    det = _make(False, orig, flip)
    r = det.detect(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10)), object())
    assert np.array_equal(r.xy_det, orig)               # xy = structure(xy_raw) = orig (identity)


def test_perfect_flip_agreement_gives_max_conf_and_no_coord_change():
    orig = _base_orig()
    flip = _flip_of(orig)                                # aligns exactly onto orig
    det = _make(True, orig, flip)
    r = det.detect(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10)), object())
    assert np.allclose(r.xy_det, orig, atol=1e-9)        # fuse(orig, orig) = orig
    assert r.confidence >= 0.99                          # zero flip disagreement → conf≈1


def test_small_disagreement_is_averaged():
    orig = _base_orig()
    flip = _flip_of(orig)
    # nudge the flipped detection of point 0's sibling so aligned[0] = orig[0] + (4,0) (< guard 8)
    swap0 = LUNG_INDEX["L1"]                             # sibling of index 0 (R1)
    flip[swap0, 0] -= 4.0                                # aligned[0,0] = 517-(flip-4)=orig+4
    det = _make(True, orig, flip)
    r = det.detect(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10)), object())
    assert abs(r.xy_det[0, 0] - (orig[0, 0] + 2.0)) < 1e-6     # averaged: orig + 4/2


def test_large_disagreement_keeps_original():
    orig = _base_orig()
    flip = _flip_of(orig)
    swap0 = LUNG_INDEX["L1"]
    flip[swap0, 0] -= 40.0                               # aligned[0] = orig + 40 (> guard 8) → keep
    det = _make(True, orig, flip)
    r = det.detect(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10)), object())
    assert abs(r.xy_det[0, 0] - orig[0, 0]) < 1e-6      # guard → original kept, not averaged
