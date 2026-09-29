import importlib.util
import json
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('sim.auscultation_skill_v1.rgbd') is not None, 'RGB-D interface is missing'
    from sim.auscultation_skill_v1 import rgbd
    return rgbd


K = dict(fx=1200., fy=1100., cx=960., cy=540., width=1920, height=1080)


def test_optical_projection_roundtrip_and_scaled_resolution():
    rgbd = module()
    transform = np.eye(4); transform[:3, 3] = [1., -.5, .7]
    points = np.array([[1., -.5, 1.7], [1.1, -.55, 1.2]])
    uv, z = rgbd.project_points(points, K, transform)
    np.testing.assert_allclose(rgbd.backproject(uv, z, K, transform), points, atol=1e-12)
    smaller = rgbd.scale_intrinsics(K, 960, 540)
    uv2, z2 = rgbd.project_points(points, smaller, transform)
    np.testing.assert_allclose(uv2, uv/2.)
    np.testing.assert_allclose(z2, z)


def test_approved_camera_rotation_matches_saved_target_projections():
    rgbd = module()
    camera = json.loads(rgbd.CAMERA_CONFIG.read_text())
    transform = np.array(camera['world_from_optical'])
    optical = np.array([row['optical_xyz_m'] for row in camera['projected_targets']])
    world = (transform @ np.column_stack([optical, np.ones(len(optical))]).T).T[:, :3]
    uv, z = rgbd.project_points(world, camera['intrinsics'], transform)
    np.testing.assert_allclose(uv, [row['uv_px'] for row in camera['projected_targets']], atol=1e-9)
    np.testing.assert_allclose(z, optical[:, 2], atol=1e-12)
    np.testing.assert_allclose(rgbd.backproject(uv, z, camera['intrinsics'], transform), world, atol=1e-12)


def test_depth_report_retains_occlusion_and_invalid_depth():
    rgbd = module()
    intrinsics = dict(fx=100., fy=100., cx=2., cy=2., width=5, height=5)
    depth = np.ones((5, 5), np.float32)
    depth[2, 2] = .7
    records = rgbd.target_depth_report({'center': [0., 0., 1.], 'behind': [0., 0., -1.]}, depth, intrinsics, np.eye(4))
    assert records[0]['occluded_by_10mm'] is True
    assert records[0]['depth_residual_m'] == pytest.approx(-.3)
    assert records[1]['inside'] is False and records[1]['measured_depth_m'] is None
    depth[2, 2] = np.inf
    assert rgbd.target_depth_report({'center': [0., 0., 1.]}, depth, intrinsics, np.eye(4))[0]['valid_depth'] is False


def test_schedule_uses_actual_saved_rows_for_each_required_state():
    rgbd = module()
    rows = []
    for i, subphase in enumerate(['settle', 'align', 'contact_cycle', 'contact_cycle', 'normal_release_confirm', 'home_settle']):
        rows.append(dict(target='L4', phase='HOLD' if i in (2, 3) else 'DONE' if i>=4 else 'APPROACH',
                         subphase=subphase, time_s=(i+1)/240.))
    samples = rgbd.select_samples(rows, np.arange(6), ['L4'])
    assert [s['kind'] for s in samples] == ['initial_home', 'prepose', 'hold', 'release', 'final_home']
    assert all(s['physics_row'] == s['pose_index'] for s in samples)
    with pytest.raises(ValueError, match='HOLD'):
        rgbd.select_samples(rows, np.array([0, 1, 4, 5]), ['L4'])


def test_projection_rejects_invalid_calibration_and_depth():
    rgbd = module()
    invalid = np.eye(4); invalid[0, 0] = 2.
    with pytest.raises(ValueError):
        rgbd.project_points([[0., 0., 1.]], K, invalid)
    with pytest.raises(ValueError):
        rgbd.backproject([[960., 540.]], [-1.], K, np.eye(4))
