"""Numerical placement checks use metres and actual face winding."""
import importlib
import importlib.util
import hashlib
import json

import numpy as np
import pytest


def placement_module():
    name = 'sim.ausc_policy.body_placement'
    assert importlib.util.find_spec(name), 'Numerical body placement module is missing'
    return importlib.import_module(name)


def fixture_body():
    # SMPL anatomical left/up/front = X/Y/Z. Bent thighs point forward.
    joints = np.zeros((24, 3))
    joints[1:3] = [[.1, 0, 0], [-.1, 0, 0]]
    joints[3] = [0, .15, 0]
    joints[4:6] = [[.1, -.05, .4], [-.1, -.05, .4]]
    joints[6] = [0, .3, 0]
    joints[7:9] = [[.1, -.4, .4], [-.1, -.4, .4]]
    joints[9] = [0, .45, 0]
    joints[10:12] = [[.1, -.43, .52], [-.1, -.43, .52]]
    joints[12] = [0, .6, 0]
    joints[13:15] = [[.1, .5, 0], [-.1, .5, 0]]
    joints[15] = [0, .7, 0]
    joints[16:18] = [[.2, .5, 0], [-.2, .5, 0]]
    joints[18:20] = [[.3, .3, 0], [-.3, .3, 0]]
    joints[20:22] = [[.25, .1, .1], [-.25, .1, .1]]
    joints[22:24] = [[.25, .05, .1], [-.25, .05, .1]]
    vertices = np.array([
        [-.12, -.1, -.08], [.12, -.1, -.08], [0, -.1, .08],
        [.08, -.45, .5], [.12, -.45, .5], [.1, -.43, .55],
        [-.12, -.45, .5], [-.08, -.45, .5], [-.1, -.43, .55],
        [-.08, .3, -.1], [0, .4, -.1], [.08, .3, -.1],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    inventory = {f'{side}{level}': dict(smpl_vertex_indices=[11, 9, 10],
                    barycentric_weights=[.2, .3, .5])
                 for level in range(1, 6) for side in ('L', 'R')}
    from sim.ausc_policy.surface_anchors import evaluate_anchors
    anchors = evaluate_anchors(vertices, faces, inventory)
    body = dict(vertices=vertices, faces=faces, joints=joints,
                anchor_names=np.array(list(anchors)),
                anchor_positions=np.array([v['position_m'] for v in anchors.values()]),
                anchor_normals=np.array([v['normal'] for v in anchors.values()]))
    return body, inventory


def fixture_seat(**overrides):
    module = placement_module()
    config = dict(center_world_m=(0., 1., .5), up_world=(0., 0., 1.),
                  front_world=(0., 1., 0.), half_extents_m=(.17, .17),
                  thickness_m=.04, floor_z_m=0., geometry_tolerance_m=.003,
                  table_bounds_world_m=((-1., -.5, 0.), (1., .5, .4)),
                  camera_world_from_optical=((1., 0., 0., 0.), (0., 0., 1., 0.),
                                             (0., -1., 0., .8), (0., 0., 0., 1.)),
                  camera_intrinsics=dict(fx=500., fy=500., cx=500., cy=500., width=1000, height=1000))
    config.update(overrides)
    return module.SeatConfig(**config)


def test_rigid_placement_preserves_geometry_and_transports_winding_normals():
    module = placement_module()
    body, inventory = fixture_body()
    result = module.place_body(body, fixture_seat(), inventory)
    transform = result['world_from_smpl']
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    np.testing.assert_allclose(result['joints'][0], [0., 1., .6], atol=1e-12)
    np.testing.assert_allclose(result['anchor_normals'][0], [0., -1., 0.], atol=1e-12)
    np.testing.assert_allclose(result['vertices'], body['vertices'] @ rotation.T + transform[:3, 3])
    assert result['qa']['numerical_gate_passed']
    assert result['qa']['metrics']['seat_support_gap_m'] == pytest.approx(0., abs=1e-12)
    assert result['qa']['metrics']['foot_min_floor_clearances_m'] == pytest.approx([.15, .15])
    assert result['qa']['foot_support_status'] == 'unverified_feet_above_floor'
    assert result['qa']['training_eligible'] is False


def test_source_rotation_translation_does_not_change_world_placement():
    module = placement_module()
    body, inventory = fixture_body()
    baseline = module.place_body(body, fixture_seat(), inventory)
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = {k: v.copy() for k, v in body.items()}
    for key in ('vertices', 'joints', 'anchor_positions'):
        moved[key] = moved[key] @ rotation.T + [2., -3., 1.]
    moved['anchor_normals'] = moved['anchor_normals'] @ rotation.T
    result = module.place_body(moved, fixture_seat(), inventory)
    np.testing.assert_allclose(result['vertices'], baseline['vertices'], atol=1e-12)
    np.testing.assert_allclose(result['anchor_normals'], baseline['anchor_normals'], atol=1e-12)


@pytest.mark.parametrize('normal', [[0., 0., 0.], [0., 0., 2.], [np.nan, 0., 0.], [0., 0., 1.]])
def test_invalid_or_reversed_input_normal_is_rejected(normal):
    module = placement_module()
    body, inventory = fixture_body()
    body['anchor_normals'][0] = normal
    with pytest.raises(ValueError, match='normal'):
        module.place_body(body, fixture_seat(), inventory)


def test_degenerate_body_axes_are_rejected():
    module = placement_module()
    body, inventory = fixture_body()
    body['joints'][12] = body['joints'][0]
    with pytest.raises(ValueError, match='Degenerate'):
        module.place_body(body, fixture_seat(), inventory)


def test_standing_pose_is_not_accepted_as_seated():
    module = placement_module()
    body, inventory = fixture_body()
    body['joints'][4:6] = [[.1, -.4, 0], [-.1, -.4, 0]]
    body['joints'][7:9] = [[.1, -.8, 0], [-.1, -.8, 0]]
    result = module.place_body(body, fixture_seat(), inventory)
    assert not result['qa']['numerical_gate_passed']
    assert 'pose_not_seated' in result['qa']['rejection_reasons']


def test_floor_penetration_rejected_but_unsupported_high_stool_not_rejected():
    module = placement_module()
    body, inventory = fixture_body()
    result = module.place_body(body, fixture_seat(center_world_m=(0., 1., .2)), inventory)
    assert 'floor_penetration' in result['qa']['rejection_reasons']


def test_table_overlap_is_conservative_rejection_not_collision_claim():
    module = placement_module()
    body, inventory = fixture_body()
    seat = fixture_seat(table_bounds_world_m=((-1., .8, .1), (1., 1.7, 1.)))
    result = module.place_body(body, seat, inventory)
    assert 'table_aabb_overlap_unresolved' in result['qa']['rejection_reasons']
    assert result['qa']['table_collision_status'] == 'unverified_conservative_aabb_overlap'


def test_collinear_seat_axes_and_unknown_clothing_are_rejected():
    module = placement_module()
    body, inventory = fixture_body()
    with pytest.raises(ValueError, match='orthogonal'):
        module.place_body(body, fixture_seat(front_world=(0., 0., 1.)), inventory)
    with pytest.raises(ValueError, match='clothing'):
        module.place_body(body, fixture_seat(), inventory, clothing='existing_shirt')


def test_reversed_patient_front_fails_camera_back_orientation():
    module = placement_module()
    body, inventory = fixture_body()
    result = module.place_body(body, fixture_seat(front_world=(0., -1., 0.)), inventory)
    assert 'back_not_facing_camera' in result['qa']['rejection_reasons']


def test_missing_target_or_changed_surface_position_is_rejected():
    module = placement_module()
    body, inventory = fixture_body()
    body['anchor_positions'][0, 0] += .02
    with pytest.raises(ValueError, match='position'):
        module.place_body(body, fixture_seat(), inventory)
    body, inventory = fixture_body()
    body['anchor_names'][0] = 'L0'
    with pytest.raises(ValueError, match='ten V6A'):
        module.place_body(body, fixture_seat(), inventory)


def scene_fixture():
    identity = np.eye(4).tolist()
    paths_bounds = [
        ('/World/GroundPlane/geom', [[-10, -10, 0], [10, 10, 0]]),
        ('/World/Stool/seat', [[-.17, .83, .46], [.17, 1.17, .5]]),
        ('/World/Table/table_low', [[-1, -.5, 0], [1, .5, .4]]),
        ('/World/Patient/sneakers', [[-.15, 1.3, .15], [.15, 1.7, .25]]),
        ('/World/Patient/scrubsshirt', [[-.2, .8, .6], [.2, 1.2, 1.1]]),
    ]
    geometry = dict(colliders=[dict(path=p, bounds_local=b, world_from_body=identity)
                              for p, b in paths_bounds])
    seat = fixture_seat()
    camera = dict(render_variant='up_5cm_down_7deg', world_from_optical=seat.camera_world_from_optical,
                  intrinsics=seat.camera_intrinsics)
    return geometry, camera


def test_scene_config_reads_transformed_geometry_and_separate_current_camera():
    module = placement_module()
    geometry, camera = scene_fixture()
    geometry['colliders'][1]['world_from_body'] = np.eye(4).tolist()
    geometry['colliders'][1]['world_from_body'][0][3] = .01
    config, reference = module.seat_config_from_scene(geometry, camera,
        front_world=(0, 1, 0), geometry_tolerance_m=.003)
    np.testing.assert_allclose(config.center_world_m, [.01, 1, .5])
    assert reference['nominal_shoe_min_floor_clearance_m'] == .15
    assert reference['existing_shirt_status'] == 'nominal_reference_only_not_fitted'
    np.testing.assert_allclose(config.camera_world_from_optical, camera['world_from_optical'])


def test_pilot_preserves_sources_hashes_split_and_rejects_hash_change(tmp_path):
    module = placement_module()
    body, anchors = fixture_body()
    source = tmp_path / 'source'
    source.mkdir()
    asset = source / 'body00_seated_lean_asset.npz'
    np.savez_compressed(asset, **body)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps(dict(anchors=anchors)))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(splits=dict(train=['fixture_identity'], val=[], test=[]))))
    manifest = [dict(identity='fixture_identity', pose='seated_lean_asset', file=asset.name, sha256=sha(asset))]
    (source / 'asset_manifest.json').write_text(json.dumps(manifest))
    (source / 'summary.json').write_text(json.dumps(dict(identities=1, identity_split_sha256=sha(split),
        anchor_inventory_sha256=sha(inventory), smpl_neutral_sha256='fixture_model_hash')))
    geometry, camera = scene_fixture()
    geo_path = tmp_path / 'geometry.json'
    geo_path.write_text(json.dumps(geometry))
    camera_path = tmp_path / 'camera.json'
    camera_path.write_text(json.dumps(camera))
    kwargs = dict(source_dir=source, inventory_path=inventory, identity_split_path=split,
                  geometry_path=geo_path, camera_path=camera_path, front_world=(0, 1, 0),
                  geometry_tolerance_m=.003)
    summary = module.run_pilot(output_dir=tmp_path / 'placed', **kwargs)
    assert summary['placed_identities'] == 1
    assert summary['numerical_gate_passed_count'] == 1
    assert summary['training_eligible'] is False
    placed_manifest = json.loads((tmp_path / 'placed' / 'asset_manifest.json').read_text())
    assert placed_manifest[0]['source_sha256'] == manifest[0]['sha256'] == sha(asset)
    with np.load(tmp_path / 'placed' / placed_manifest[0]['file']) as placed:
        assert placed['vertices'].shape == body['vertices'].shape
        assert 'world_from_smpl' in placed
    asset.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        module.run_pilot(output_dir=tmp_path / 'bad', **kwargs)
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('overlap_m', [0., .001, .003])
def test_table_contact_or_overlap_within_tolerance_does_not_claim_separation(overlap_m):
    module = placement_module()
    body, inventory = fixture_body()
    baseline = module.place_body(body, fixture_seat(), inventory)
    body_min_y = baseline['vertices'][:, 1].min()
    seat = fixture_seat(table_bounds_world_m=((-1., -.5, 0.), (1., body_min_y + overlap_m, 1.)))
    result = module.place_body(body, seat, inventory)
    assert result['qa']['metrics']['table_aabb_separation_lower_bound_m'] == 0.
    assert result['qa']['table_collision_status'] == 'unverified_conservative_aabb_overlap'
    assert 'table_aabb_overlap_unresolved' in result['qa']['rejection_reasons']
    assert not result['qa']['numerical_gate_passed']


@pytest.mark.parametrize('axis', ['up_world', 'front_world'])
def test_seat_axes_fail_strict_unit_check_for_small_scale_errors(axis):
    module = placement_module()
    body, inventory = fixture_body()
    value = np.array([0., 0., 1.] if axis == 'up_world' else [0., 1., 0.]) * 1.000001
    with pytest.raises(ValueError, match='unit and orthogonal'):
        module.place_body(body, fixture_seat(**{axis: tuple(value)}), inventory)


def test_tolerated_axis_roundoff_is_removed_before_rigid_transform():
    module = placement_module()
    body, inventory = fixture_body()
    seat = fixture_seat(up_world=(0., 0., 1. + 2e-11), front_world=(0., 1. - 2e-11, 1e-12))
    result = module.place_body(body, seat, inventory)
    rotation = result['world_from_smpl'][:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15, rtol=0.)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-15, rel=0.)


def test_scene_loader_rejects_tilted_stool_before_horizontal_aabb_placement():
    module = placement_module()
    geometry, camera = scene_fixture()
    angle = np.deg2rad(10.)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                     [0., np.sin(angle), np.cos(angle)]]
    geometry['colliders'][1]['world_from_body'] = matrix.tolist()
    with pytest.raises(ValueError, match='tilted stool'):
        module.seat_config_from_scene(geometry, camera, front_world=(0, 1, 0), geometry_tolerance_m=.003)
