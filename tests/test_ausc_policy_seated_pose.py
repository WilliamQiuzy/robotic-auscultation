"""Synthetic pose provenance and numeric arm intersection regression tests."""
import importlib
import importlib.util
import hashlib
import json

import numpy as np
import pytest


def module():
    name = 'sim.ausc_policy.seated_pose'
    assert importlib.util.find_spec(name), 'Synthetic seated pose module is missing'
    return importlib.import_module(name)


def test_proposals_change_only_explicit_arm_rotations_from_source_seated_lean():
    m = module()
    proposals = m.synthetic_pose_proposals()
    assert len(proposals) == 3
    assert len({p['name'] for p in proposals}) == 3
    baseline = np.zeros((23, 3), np.float32)
    baseline[0, 0] = baseline[1, 0] = -1.25
    baseline[3, 0] = baseline[4, 0] = 1.4
    baseline[2, 0] = .12
    for proposal in proposals:
        pose = np.asarray(proposal['body_pose_axis_angle_rad']).reshape(23, 3)
        np.testing.assert_array_equal(pose[:15], baseline[:15])
        assert pose[15, 2] < -1.
        assert pose[16, 2] > 1.
        assert pose[15, 1] == pytest.approx(-np.deg2rad(15.))
        assert pose[16, 1] == pytest.approx(np.deg2rad(15.))
        assert abs(pose[15, 2]) == pytest.approx(np.deg2rad(60.))
        assert pose[17, 1] < 0.
        assert pose[18, 1] > 0.
        assert proposal['pose_source'] == 'authored_synthetic_joint_rotations_not_HuMMan_motion'
        assert proposal['mesh_scale'] == 1.
        assert proposal['patient_posture_qualified'] is False


def test_nonadjacent_crossing_triangles_are_detected_without_fcl():
    m = module()
    vertices = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                         [0., 0., -1.], [0., 0., 1.], [.5, 0., .2]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    result = m.triangle_intersections(vertices, faces, [0], [1])
    assert result['intersecting_pair_count'] == 1
    assert result['sample_face_pairs'] == [[0, 1]]
    assert result['physics_verified'] is False


def test_separated_triangles_and_shared_surface_seams_do_not_count_as_crossings():
    m = module()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5], [1, 6, 2]])
    assert m.triangle_intersections(vertices, faces, [0], [1])['intersecting_pair_count'] == 0
    result = m.triangle_intersections(vertices, faces, [0], [2])
    assert result['intersecting_pair_count'] == 0
    assert result['excluded_shared_vertex_pair_count'] == 1


def test_coplanar_candidate_is_unresolved_instead_of_clear():
    m = module()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [.1, .1, 0.], [.5, .1, 0.], [.1, .5, 0.]])
    result = m.triangle_intersections(vertices, np.array([[0, 1, 2], [3, 4, 5]]), [0], [1])
    assert result['coplanar_unresolved_pair_count'] == 1
    assert result['numeric_intersection_screen_clear'] is False


def test_degenerate_triangle_is_invalid_not_silent_clear():
    m = module()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                         [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    with pytest.raises(ValueError, match='Degenerate'):
        m.triangle_intersections(vertices, np.array([[0, 1, 2], [3, 4, 5]]), [0], [1])


def arm_joints():
    joints = np.zeros((24, 3))
    joints[1], joints[2], joints[12] = [.1, 0, 0], [-.1, 0, 0], [0, .6, 0]
    for sign, shoulder, elbow, wrist, hand in [(1, 16, 18, 20, 22), (-1, 17, 19, 21, 23)]:
        joints[shoulder] = [sign * .2, .5, 0]
        joints[elbow] = [sign * .2, .2, 0]
        joints[wrist] = [sign * .2, 0, .2]
        joints[hand] = [sign * .2, -.03, .24]
    return joints


def test_arm_joint_screen_distinguishes_arms_down_from_t_pose():
    m = module()
    joints = arm_joints()
    metrics = m.arm_joint_metrics(joints)
    assert metrics['arms_down_numeric_screen_passed']
    np.testing.assert_allclose(metrics['upper_arm_downward_angle_deg'], [90., 90.])
    joints[18] = [.5, .5, 0]
    joints[19] = [-.5, .5, 0]
    metrics = m.arm_joint_metrics(joints)
    assert not metrics['arms_down_numeric_screen_passed']
    np.testing.assert_allclose(metrics['upper_arm_downward_angle_deg'], [0., 0.])


def test_lbs_arm_geometry_regions_are_explicit_unsigned_mesh_screens():
    m = module()
    vertices = np.array([[.5, 0., 0.], [.7, 0., 0.], [.5, .2, 0.],
                         [-.7, 0., 0.], [-.5, 0., 0.], [-.5, .2, 0.],
                         [-.2, 0., 0.], [.2, 0., 0.], [0., .2, 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    weights = np.zeros((9, 24))
    weights[:3, 18] = weights[3:6, 19] = weights[6:, 1] = 1.
    result = m.evaluate_arm_geometry(vertices, faces, arm_joints(), weights)
    assert result['numeric_arm_screen_passed']
    assert result['region_method'] == 'dominant_SMPL_LBS_joint_majority_face_proxy'
    assert result['distance_status'] == 'unsigned_open_region_distance_not_penetration_or_support'
    assert result['patient_posture_qualified'] is False
    assert result['distal_arm_to_nonarm_surface_min_m'] > 0.
    assert result['comparisons']['left_arm_vs_thigh']['first_region_face_count'] == 1


@pytest.mark.parametrize('invalid', ['model_hash', 'heldout_identity', 'asset_hash', 'split_hash', 'beta_nan'])
def test_pilot_rejects_changed_sources_before_creating_outputs(tmp_path, invalid):
    m = module()
    source, identity = tmp_path / 'source', tmp_path / 'identity'
    source.mkdir()
    identity.mkdir()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    model_path = tmp_path / 'model.pkl'
    model_path.write_bytes(b'fixture_model_preflight_only')
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'anchors': {}}))
    ids = [f'fixture{i}' for i in range(8)]
    split = dict(splits=dict(train=ids if invalid != 'heldout_identity' else ids[:-1],
                             val=[] if invalid != 'heldout_identity' else ids[-1:], test=[]))
    (identity / 'identity_split.json').write_text(json.dumps(split))
    split_sha = sha(identity / 'identity_split.json')
    betas = {name: [0.] * 10 for name in ids}
    if invalid == 'beta_nan':
        betas[ids[0]][0] = float('nan')
    (identity / 'identity_quality.json').write_text(json.dumps(dict(
        identity_split_sha256=split_sha, eligible_shape_betas=betas)))
    records = []
    for index, name in enumerate(ids):
        file = f'body{index:02d}_seated_lean_asset.npz'
        (source / file).write_bytes(b'fixture_asset_preflight_only')
        records.append(dict(identity=name, pose='seated_lean_asset', file=file, sha256=sha(source / file)))
    (source / 'asset_manifest.json').write_text(json.dumps(records))
    (source / 'summary.json').write_text(json.dumps(dict(
        identities=8, identity_split_sha256=split_sha if invalid != 'split_hash' else 'changed',
        anchor_inventory_sha256=sha(inventory), smpl_neutral_sha256=sha(model_path))))
    if invalid == 'model_hash':
        model_path.write_bytes(b'changed_model')
    if invalid == 'asset_hash':
        (source / records[0]['file']).write_bytes(b'changed_asset')
    seat_path = tmp_path / 'seat.json'
    seat_path.write_text('{}')
    with pytest.raises(ValueError):
        m.run_pilot(source_dir=source, identity_dir=identity, inventory_path=inventory,
                    model_path=model_path, seat_config_path=seat_path, output_dir=tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
