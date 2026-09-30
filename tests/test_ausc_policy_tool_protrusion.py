import numpy as np
import pytest

from sim.ausc_policy import tool_protrusion as tool


def test_contact_face_is_two_mm_ahead_and_tcp_follows_compression():
    assert tool.FACE_Z_M == pytest.approx(-.044)
    assert tool.LIP_Z_M - tool.FACE_Z_M == pytest.approx(.002)
    assert tool.wrist_contact_offset(.003) == pytest.approx(.126)
    with pytest.raises(ValueError):
        tool.wrist_contact_offset(.01)


def test_whole_head_translation_preserves_shape():
    v = np.array([[.023, 0, -.0412], [-.023, 0, -.0352]])
    old = v.copy()
    shifted = tool.shift_head(v)
    np.testing.assert_allclose(shifted[:, :2], old[:, :2])
    np.testing.assert_allclose(shifted[:, 2], old[:, 2] - .0028)
    np.testing.assert_array_equal(v, old)


def test_retaining_opening_clears_head_without_changing_outer_envelope():
    v = np.array([[.021, 0, -.042], [.021, 0, -.0412],
                  [.0265, 0, -.042], [.0235, 0, -.035], [.01, 0, -.03]])
    old = v.copy()
    corrected = tool.open_lower_bore(v)
    np.testing.assert_allclose(corrected[:2, 0], .0235)
    np.testing.assert_array_equal(corrected[2:], old[2:])
    np.testing.assert_array_equal(corrected[:, 2], old[:, 2])
    np.testing.assert_array_equal(v, old)


def test_bad_geometry_is_rejected():
    with pytest.raises(ValueError):
        tool.shift_head([[0., float('nan'), 0.]])
    with pytest.raises(ValueError):
        tool.open_lower_bore([[0., 0., -.042]])
