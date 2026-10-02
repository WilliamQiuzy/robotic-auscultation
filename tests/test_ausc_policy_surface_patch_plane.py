import importlib

import numpy as np
import pytest


def fit(points, normal=(0, 0, 1), limit=20.):
    module = importlib.import_module('sim.ausc_policy.surface_patch_plane')
    return module.fit_height_envelope(points, normal, slope_component_limit_deg=limit)


def grid():
    x, y = np.meshgrid(np.linspace(-.023, .023, 5), np.linspace(-.023, .023, 5))
    return x.ravel(), y.ravel()


def test_tilted_plane_has_zero_residual_span_without_moving_surface():
    x, y = grid()
    points = np.c_[x, y, .1*x + .2*y + .003]
    original = points.copy()
    report = fit(points)
    assert report['residual_span_m'] == pytest.approx(0., abs=1e-9)
    assert report['normal_world'] == pytest.approx(np.array([-.1, -.2, 1])/np.sqrt(1.05))
    assert report['normal_angle_deg'] == pytest.approx(np.degrees(np.arctan(np.sqrt(.05))))
    assert np.array_equal(points, original)
    assert report['training_eligible'] is False


def test_symmetric_curvature_cannot_be_removed_by_tilt():
    x, y = grid()
    z = (x*x + y*y)/(2*.3)
    report = fit(np.c_[x, y, z])
    assert report['residual_span_m'] == pytest.approx(z.max()-z.min(), abs=1e-9)


def test_rigid_motion_preserves_height_span():
    from scipy.spatial.transform import Rotation
    x, y = grid()
    points = np.c_[x, y, (x*x+y*y)/.6]
    rotation = Rotation.from_euler('xyz', [.1, .5, .7]).as_matrix()
    first = fit(points)
    second = fit(points @ rotation.T + [1, 2, 3], rotation[:, 2])
    assert second['residual_span_m'] == pytest.approx(first['residual_span_m'], abs=1e-9)


def test_slope_bounds_do_not_silently_flatten_steep_plane():
    x, y = grid()
    report = fit(np.c_[x, y, x], limit=10.)
    assert report['residual_span_m'] > .03
    assert max(abs(x) for x in report['slopes']) <= np.tan(np.deg2rad(10.)) + 1e-9


@pytest.mark.parametrize('points,normal', [
    ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, 0, 1]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, float('nan')]], [0, 0, 1]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 2]),
    ([[0, 0, 0]], [0, 0, 1]),
])
def test_missing_degenerate_or_invalid_surface_rejected(points, normal):
    with pytest.raises(ValueError):
        fit(points, normal)


def test_surface_probe_has_complete_flat_footprint():
    module = importlib.import_module('sim.ausc_policy.surface_patch_plane')
    triangles = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
                          [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], float)
    report = module.probe_surface_patch(triangles, [0, 0, 0], [0, 0, 1])
    assert report['requested_samples'] == 97
    assert report['missing_samples'] == 0
    assert report['plane']['residual_span_m'] == pytest.approx(0, abs=1e-9)
    assert report['ring_plane']['residual_span_m'] == pytest.approx(0, abs=1e-9)


def test_missing_surface_has_no_fitted_plane():
    module = importlib.import_module('sim.ausc_policy.surface_patch_plane')
    report = module.probe_surface_patch(np.empty((0, 3, 3)), [0, 0, 0], [0, 0, 1])
    assert report['missing_samples'] == 97
    assert report['plane'] is None
    assert report['ring_plane'] is None


def test_outward_offset_preserves_topology_and_records_nonphysical_scope():
    module = importlib.import_module('sim.ausc_policy.surface_patch_plane')
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    shifted, receipt = module.offset_surface_proxy(vertices, faces, .0005)
    assert np.linalg.norm(shifted - vertices, axis=1) == pytest.approx(np.full(6, .0005))
    assert shifted == pytest.approx(vertices*1.0005)
    assert receipt['calibrated_cloth'] is False
    assert receipt['training_eligible'] is False


def test_offset_rejects_inward_winding():
    module = importlib.import_module('sim.ausc_policy.surface_patch_plane')
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])[:, ::-1]
    with pytest.raises(ValueError, match='winding'):
        module.offset_surface_proxy(vertices, faces, .0005)
