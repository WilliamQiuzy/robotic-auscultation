"""Crop stabilization (2026-08-07 audit item #1) + byte-identity guard.

The stabilizer damps the per-frame crop-box flicker that is the dominant common-mode 3D jitter
source, but ONLY when opted in — ``back_square_crop`` with default args must stay byte-identical
to the pre-audit min/max behaviour (the live detector must not shift under my feet).
"""
import numpy as np

from ausc.perception.tracking.geometry import (
    CropStabilizer,
    Intrinsics,
    back_square_crop,
)


# ── CropStabilizer state machine ─────────────────────────────────────────────
def test_first_frame_is_raw():
    s = CropStabilizer()
    assert s.update(100.0, 200.0, 300.0) == (100.0, 200.0, 300.0)


def test_deadband_holds_small_flicker():
    s = CropStabilizer(deadband_frac=0.02)              # deadband = 2% of side=300 = 6 px
    s.update(100.0, 200.0, 300.0)
    cx, cy, side = s.update(103.0, 200.0, 300.0)        # 3 px < 6 → held
    assert (cx, side) == (100.0, 300.0)


def test_snap_on_large_motion():
    s = CropStabilizer(snap_frac=0.25)                  # snap above 25% of side=300 = 75 px
    s.update(100.0, 200.0, 300.0)
    cx, _, _ = s.update(200.0, 200.0, 300.0)            # 100 px > 75 → snap through, no lag
    assert cx == 200.0


def test_ema_between_deadband_and_snap():
    s = CropStabilizer(ema=0.5, deadband_frac=0.02, snap_frac=0.25)
    s.update(100.0, 200.0, 300.0)
    cx, _, _ = s.update(130.0, 200.0, 300.0)            # 30 px: >6, <75 → EMA 100 + 0.5*30
    assert abs(cx - 115.0) < 1e-9


def test_reset_returns_to_raw():
    s = CropStabilizer()
    s.update(100.0, 200.0, 300.0)
    s.reset()
    assert s.update(400.0, 500.0, 600.0) == (400.0, 500.0, 600.0)


# ── back_square_crop byte-identity when not opted in ─────────────────────────
def _fake_scene():
    # a slab of depth ~1 m over the central region → a well-defined body-band mask
    depth = np.zeros((480, 640), np.float32)
    depth[120:360, 200:440] = 1000.0                    # mm
    intr = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, 1.0)
    rgb = np.zeros((480, 640, 3), np.uint8)
    return rgb, depth, intr


def test_default_crop_is_unchanged_by_new_params():
    rgb, depth, intr = _fake_scene()
    a_img, a = back_square_crop(rgb, depth, intr)
    b_img, b = back_square_crop(rgb, depth, intr, pct=None, stabilizer=None)
    assert (a.x0, a.y0, a.side, a.res) == (b.x0, b.y0, b.side, b.res)
    assert np.array_equal(a_img, b_img)


def test_percentile_box_is_no_larger_than_minmax():
    rgb, depth, intr = _fake_scene()
    _, mm = back_square_crop(rgb, depth, intr)
    _, pc = back_square_crop(rgb, depth, intr, pct=2.0)
    assert pc.side <= mm.side + 1                        # percentile trims flicker → not larger


def test_stabilizer_first_frame_matches_percentile_box():
    rgb, depth, intr = _fake_scene()
    stab = CropStabilizer()
    _, s0 = back_square_crop(rgb, depth, intr, pct=2.0, stabilizer=stab)
    _, p0 = back_square_crop(rgb, depth, intr, pct=2.0)
    assert (s0.x0, s0.y0, s0.side) == (p0.x0, p0.y0, p0.side)   # first frame = raw percentile box
