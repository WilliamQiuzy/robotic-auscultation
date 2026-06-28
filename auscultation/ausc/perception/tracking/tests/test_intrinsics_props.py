import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import json
import math

import pytest

from ausc.perception.tracking.intrinsics import (
    default_intrinsics,
    orbbec_intrinsics,
    save_intrinsics,
    load_intrinsics,
)
from ausc.perception.tracking.geometry import Intrinsics


# ── default_intrinsics: pinhole invariants ──────────────────────────────────
def test_default_intrinsics_centered_and_square():
    """Principal point is the image centre and pixels are square (fx == fy)."""
    intr = default_intrinsics(640, 480, hfov_deg=60.0)
    assert intr.cx == pytest.approx(320.0)
    assert intr.cy == pytest.approx(240.0)
    assert intr.fx == pytest.approx(intr.fy)
    assert intr.width == 640 and intr.height == 480


def test_default_intrinsics_focal_matches_fov():
    """fx must satisfy hfov = 2*atan((W/2)/fx) for the requested horizontal FOV."""
    w, h, hfov = 1280, 720, 70.0
    intr = default_intrinsics(w, h, hfov_deg=hfov)
    recovered = math.degrees(2.0 * math.atan((w / 2.0) / intr.fx))
    assert recovered == pytest.approx(hfov, rel=1e-9)


def test_default_intrinsics_positive_and_finite():
    """A sane FOV yields strictly positive, finite focal lengths."""
    intr = default_intrinsics(800, 600, hfov_deg=55.0, depth_scale=0.5)
    for v in (intr.fx, intr.fy, intr.cx, intr.cy):
        assert np.isfinite(v) and v > 0.0
    assert intr.depth_scale == 0.5


def test_default_intrinsics_dimension_types():
    """width/height stay plain ints (used for array indexing downstream)."""
    intr = default_intrinsics(640, 480)
    assert isinstance(intr.width, int)
    assert isinstance(intr.height, int)


def test_default_intrinsics_narrower_fov_longer_focal():
    """Monotonicity: a narrower FOV must give a longer focal length."""
    wide = default_intrinsics(640, 480, hfov_deg=90.0)
    narrow = default_intrinsics(640, 480, hfov_deg=40.0)
    assert narrow.fx > wide.fx


# ── save / load round-trip ──────────────────────────────────────────────────
def _assert_intr_equal(a: Intrinsics, b: Intrinsics):
    assert a.fx == pytest.approx(b.fx)
    assert a.fy == pytest.approx(b.fy)
    assert a.cx == pytest.approx(b.cx)
    assert a.cy == pytest.approx(b.cy)
    assert a.width == b.width
    assert a.height == b.height
    assert a.depth_scale == pytest.approx(b.depth_scale)


def test_save_load_roundtrip_all_fields(tmp_path):
    """Every field (including a non-default depth_scale) survives a write/read cycle."""
    intr = Intrinsics(fx=525.3, fy=524.7, cx=319.5, cy=239.5,
                      width=640, height=480, depth_scale=0.25)
    p = str(tmp_path / "calib.json")
    save_intrinsics(p, intr)
    loaded = load_intrinsics(p)
    _assert_intr_equal(loaded, intr)


def test_save_load_roundtrip_default_intrinsics(tmp_path):
    """A default-derived calibration round-trips bit-for-bit on floats."""
    intr = default_intrinsics(1280, 720, hfov_deg=65.0, depth_scale=1.0)
    p = str(tmp_path / "default.json")
    save_intrinsics(p, intr)
    loaded = load_intrinsics(p)
    _assert_intr_equal(loaded, intr)
    # focal length is irrational; require near-exact float preservation via JSON.
    assert loaded.fx == pytest.approx(intr.fx, rel=1e-12)


def test_saved_file_is_valid_json_with_expected_keys(tmp_path):
    """On-disk format is a flat JSON object with all seven documented keys."""
    intr = default_intrinsics(640, 480, depth_scale=2.0)
    p = str(tmp_path / "c.json")
    save_intrinsics(p, intr)
    with open(p) as f:
        d = json.load(f)
    assert set(d.keys()) == {"fx", "fy", "cx", "cy", "width", "height", "depth_scale"}
    assert d["depth_scale"] == 2.0


def test_load_defaults_depth_scale_when_absent(tmp_path):
    """Legacy files lacking depth_scale must load with the 1.0 default, not crash."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"fx": 500.0, "fy": 500.0, "cx": 320.0,
                             "cy": 240.0, "width": 640, "height": 480}))
    loaded = load_intrinsics(str(p))
    assert loaded.depth_scale == 1.0
    assert loaded.fx == 500.0


# ── adversarial loads ───────────────────────────────────────────────────────
def test_load_missing_file_raises(tmp_path):
    """A non-existent path must raise (FileNotFoundError), never return junk."""
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        load_intrinsics(missing)


def test_load_garbage_raises_valueerror(tmp_path):
    """Malformed JSON must surface as a ValueError (JSONDecodeError), not silent."""
    p = tmp_path / "garbage.json"
    p.write_text("this is not { valid json ]]]")
    with pytest.raises(ValueError):
        load_intrinsics(str(p))


def test_load_missing_required_key_raises(tmp_path):
    """A required key (fx) missing must raise KeyError, not fabricate a value."""
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"fy": 1.0, "cx": 1.0, "cy": 1.0,
                             "width": 1, "height": 1}))
    with pytest.raises(KeyError):
        load_intrinsics(str(p))


# ── scaled() helper round-trip ──────────────────────────────────────────────
def test_scaled_roundtrip_exact_ratio():
    """Down- then up-scaling by reciprocal exact ratios recovers the original."""
    intr = Intrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0,
                      width=640, height=480, depth_scale=1.0)
    half = intr.scaled(0.5, 0.5)
    back = half.scaled(2.0, 2.0)
    _assert_intr_equal(back, intr)


def test_scaled_keeps_integer_dimensions():
    """Scaled dimensions remain ints so they can index image arrays."""
    intr = default_intrinsics(640, 480)
    s = intr.scaled(0.5, 0.5)
    assert isinstance(s.width, int) and isinstance(s.height, int)
    assert s.width == 320 and s.height == 240


def test_scaled_preserves_depth_scale():
    """Resizing the image must not alter the depth unit (depth_scale)."""
    intr = Intrinsics(fx=520.0, fy=520.0, cx=320.0, cy=240.0,
                      width=640, height=480, depth_scale=0.1)
    s = intr.scaled(2.0, 2.0)
    assert s.depth_scale == pytest.approx(0.1)


def test_scaled_principal_point_consistency():
    """Scaling fx/cx by the same factor keeps the K matrix self-consistent."""
    intr = default_intrinsics(640, 480, hfov_deg=60.0)
    s = intr.scaled(0.5, 0.25)
    assert s.fx == pytest.approx(intr.fx * 0.5)
    assert s.cy == pytest.approx(intr.cy * 0.25)
    K = s.K
    assert K[0, 0] == pytest.approx(s.fx)
    assert K[1, 2] == pytest.approx(s.cy)
    assert K[2, 2] == 1.0


# ── orbbec_intrinsics: graceful degradation (no SDK / no hardware) ───────────
def test_orbbec_intrinsics_returns_none_without_sdk():
    """A bogus pipeline object must yield None, never raise, when the SDK call fails."""
    class FakePipeline:
        def get_stream_profile_list(self, *a, **k):
            raise RuntimeError("no SDK")

    assert orbbec_intrinsics(FakePipeline()) is None


def test_orbbec_intrinsics_none_on_plain_object():
    """Any object lacking the SDK API surface degrades to None."""
    assert orbbec_intrinsics(object()) is None


def test_orbbec_intrinsics_depth_scale_argument_does_not_force_value():
    """Even with a depth_scale passed, a failing pipeline still returns None (no partial obj)."""
    assert orbbec_intrinsics(object(), depth_scale=0.5) is None


# ── K matrix sanity from loaded calibration ─────────────────────────────────
def test_loaded_K_matches_fields(tmp_path):
    """The K built from a loaded calibration matches the stored fx/fy/cx/cy."""
    intr = Intrinsics(fx=611.1, fy=610.2, cx=321.0, cy=241.0,
                      width=640, height=480, depth_scale=1.0)
    p = str(tmp_path / "k.json")
    save_intrinsics(p, intr)
    K = load_intrinsics(p).K
    assert K[0, 0] == pytest.approx(611.1)
    assert K[1, 1] == pytest.approx(610.2)
    assert K[0, 2] == pytest.approx(321.0)
    assert K[1, 2] == pytest.approx(241.0)
