"""Synthetic 32-sided stool cylinders; no native support qualification."""
from copy import deepcopy
import importlib

import numpy as np
import pytest

from sim.ausc_policy.cpu_scene_variant import canonical_array_sha256, canonical_entry_sha256
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256


PARTS = ('seat', 'leg0', 'leg1', 'leg2', 'leg3', 'rung_x0', 'rung_x1', 'rung_y0', 'rung_y1')


def module():
    assert importlib.util.find_spec('sim.ausc_policy.stool_geometry') is not None, 'Stool geometry proposal is missing'
    return importlib.import_module('sim.ausc_policy.stool_geometry')


def cylinder(center, radius, length, axis):
    angles = np.arange(32)*2*np.pi/32
    points = np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
    vertices = np.zeros((66, 3))
    cross = [i for i in range(3) if i != axis]
    vertices[:32, cross] = points; vertices[32:64, cross] = points
    vertices[:32, axis] = -length/2; vertices[32:64, axis] = length/2
    vertices[64, axis] = -length/2; vertices[65, axis] = length/2
    vertices += center
    faces = []
    for i in range(32):
        j = (i+1) % 32
        faces.extend([[64, j, i], [65, i+32, j+32], [i, j, i+32], [j, j+32, i+32]])
    return vertices, np.array(faces, np.int32)


def fixture(*, local_frames=False, floor=.0):
    entries, arrays = [], {}
    cylinders = [([0., 0., .78], .25, .04, 2)]
    for x, y in ((-.18, -.18), (.18, -.18), (-.18, .18), (.18, .18)):
        cylinders.append(([x, y, .38], .02, .76, 2))
    cylinders += [([0., y, .28], .015, .36, 0) for y in (-.18, .18)]
    cylinders += [([x, 0., .44], .015, .36, 1) for x in (-.18, .18)]
    paths = ['/World/Stool/'+part for part in PARTS] + ['/World/Table/top', '/World/Robot/link', '/World/Camera/body']
    cylinders += [([2., 2., .3], .1, .2, 2)] * 3
    for i, (path, (center, radius, length, axis)) in enumerate(zip(paths, cylinders)):
        key = f'c{i:03d}'
        world, faces = cylinder(np.array(center)+[1., 2., floor], radius, length, axis)
        frame = np.eye(4)
        if local_frames:
            a, b = .17*(i+1), -.23
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            frame[:3, :3] = rz @ rx
            frame[:3, 3] = [2., -1., .5]
        vertices = np.linalg.solve(frame[:3, :3], (world-frame[:3, 3]).T).T
        arrays[key+'_v'], arrays[key+'_f'] = vertices, faces
        entries.append(dict(key=key, path=path, body='world', world_from_body=frame.tolist(),
            approximation='none', bounds_local=[vertices.min(0).tolist(), vertices.max(0).tolist()]))
    return dict(metadata=dict(colliders=entries, allowed_body_pairs=[['a', 'b']],
        analytic_planes=[dict(normal=[0, 0, 1], point=[0, 0, floor])], unsupported=[],
        camera_settings=dict(resolution=[640, 480])), arrays=arrays,
        new_seat_top_z_m=floor+.8, stool_xy_translation_m=[0., 0.], floor_z_m=floor,
        source_bindings=dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
            freshly_generated=True, collision_geometry_sha256='a'*64, collision_arrays_sha256='b'*64,
            source_run='synthetic-test-only'))


def world_vertices(bundle, index):
    entry = bundle['metadata']['colliders'][index]
    matrix = np.asarray(entry['world_from_body'])
    return bundle['arrays'][entry['key']+'_v'] @ matrix[:3, :3].T + matrix[:3, 3]


def update_bounds(args, index):
    entry = args['metadata']['colliders'][index]
    vertices = args['arrays'][entry['key']+'_v']
    entry['bounds_local'] = [vertices.min(0).tolist(), vertices.max(0).tolist()]


@pytest.mark.parametrize('local_frames', [False, True])
def test_original_height_and_zero_xy_preserve_geometry_identity(local_frames):
    args = fixture(local_frames=local_frames)
    result = module().build_stool_geometry(**args)
    for i in range(12):
        np.testing.assert_allclose(world_vertices(result, i), world_vertices(args, i), atol=1e-12, rtol=0)
    assert result['stool_height_parameterized'] is True
    for claim in ('static_support_qualified', 'ik_qualified', 'physics_qualified',
                  'training_eligible', 'native_scene_installed', 'source_artifact_hashes_verified'):
        assert result[claim] is False
    assert result['generation_report']['geometry_tolerance_m'] <= 1e-6


@pytest.mark.parametrize('top', [.48, 1.1])
@pytest.mark.parametrize('local_frames', [False, True])
def test_height_changes_fix_feet_preserve_seat_and_rungs_and_move_all_nine(top, local_frames):
    args = fixture(local_frames=local_frames, floor=.12)
    args.update(new_seat_top_z_m=top+.12, stool_xy_translation_m=[.11, -.09])
    result = module().build_stool_geometry(**args)
    old_length, new_length = .76, top-.04
    for i, part in enumerate(PARTS):
        before, after = world_vertices(args, i), world_vertices(result, i)
        np.testing.assert_allclose(after[:, :2]-before[:, :2], np.tile([.11, -.09], (66, 1)), atol=1e-12)
        if part == 'seat':
            np.testing.assert_allclose(after[:, 2]-before[:, 2], top-.8, atol=1e-12)
            assert np.ptp(after[:, 2]) == pytest.approx(.04)
        elif part.startswith('leg'):
            assert after[:, 2].min() == pytest.approx(.12, abs=1e-12)
            assert after[:, 2].max() == pytest.approx(.12+new_length, abs=1e-12)
            np.testing.assert_allclose(after[:, 2], .12+(before[:, 2]-.12)*new_length/old_length, atol=1e-12)
        else:
            fraction = ((before[:, 2].min()+before[:, 2].max())/2-.12)/old_length
            after_fraction = ((after[:, 2].min()+after[:, 2].max())/2-.12)/new_length
            assert after_fraction == pytest.approx(fraction)
            assert np.ptp(after[:, 2]) == pytest.approx(.03)
            assert result['generation_report']['parts']['/World/Stool/'+part]['center_height_fraction'] == pytest.approx(fraction)
        original, changed = args['metadata']['colliders'][i], result['metadata']['colliders'][i]
        assert changed['world_from_body'] == original['world_from_body']
        np.testing.assert_array_equal(result['arrays'][changed['key']+'_f'], args['arrays'][original['key']+'_f'])
        local = result['arrays'][changed['key']+'_v']
        np.testing.assert_array_equal(changed['bounds_local'], [local.min(0), local.max(0)])
        report = result['generation_report']['parts'][original['path']]
        np.testing.assert_allclose(report['proposal_bounds_world_m'], [after.min(0), after.max(0)], atol=1e-12)


def test_all_inputs_are_unchanged_unaliased_and_nonstool_content_is_exact():
    args = fixture(local_frames=True)
    original = deepcopy(args)
    for array in args['arrays'].values(): array.setflags(write=False)
    args['new_seat_top_z_m'] = .5
    result = module().build_stool_geometry(**args)
    assert args['metadata'] == original['metadata']
    for i in range(9, 12):
        assert result['metadata']['colliders'][i] == original['metadata']['colliders'][i]
    for key, value in args['arrays'].items():
        np.testing.assert_array_equal(value, original['arrays'][key])
        assert not np.shares_memory(value, result['arrays'][key])
        if int(key[1:4]) >= 9 or key.endswith('_f'):
            assert result['arrays'][key].dtype == value.dtype
            assert result['arrays'][key].tobytes() == value.tobytes()
    for name in ('allowed_body_pairs', 'analytic_planes', 'unsupported', 'camera_settings'):
        assert result['metadata'][name] == original['metadata'][name]
    result['metadata']['camera_settings']['resolution'][0] = 1
    result['source_bindings']['source_run'] = 'changed'
    result['arrays']['c000_v'][:] = 0
    assert args['metadata'] == original['metadata'] and args['source_bindings'] == original['source_bindings']
    np.testing.assert_array_equal(args['arrays']['c000_v'], original['arrays']['c000_v'])


def test_content_hashes_bind_actual_input_and_output_and_are_deterministic():
    args = fixture(); args['new_seat_top_z_m'] = .5
    result = module().build_stool_geometry(**args)
    assert result['content_bindings'] == module().build_stool_geometry(**args)['content_bindings']
    for source, bundle in (('source', args), ('proposal', result)):
        bound = result['content_bindings'][source]
        assert bound['metadata_sha256'] == canonical_entry_sha256(bundle['metadata'])
        for key, array in bundle['arrays'].items():
            assert bound['array_sha256'][key] == canonical_array_sha256(array)
        for entry in bundle['metadata']['colliders']:
            assert bound['entry_sha256'][entry['key']] == canonical_entry_sha256(entry)


def test_accepts_reordered_cylinder_vertices_without_changing_topology():
    args = fixture(); key = 'c000'
    order = np.random.default_rng(3).permutation(66)
    reverse = np.argsort(order)
    args['arrays'][key+'_v'] = args['arrays'][key+'_v'][order]
    args['arrays'][key+'_f'] = reverse[args['arrays'][key+'_f']]
    result = module().build_stool_geometry(**args)
    np.testing.assert_array_equal(result['arrays'][key+'_f'], args['arrays'][key+'_f'])


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'stool_root', 'missing_array',
    'vertex_nan', 'face_outside', 'face_float', 'face_degenerate', 'missing_cap', 'overlapping_faces',
    'noncircular_profile', 'rotated_world_axis', 'stale_bounds', 'bad_frame', 'reflection',
    'leg_off_floor', 'leg_off_seat', 'rung_below_floor', 'rung_above_seat'])
def test_unsupported_or_invalid_source_geometry_fails_explicitly(kind):
    args = fixture()
    if kind == 'missing':
        args['metadata']['colliders'].pop(1)
        args['arrays'].pop('c001_v'); args['arrays'].pop('c001_f')
    elif kind == 'extra': args['metadata']['colliders'][-1]['path'] = '/World/Stool/extra'
    elif kind == 'duplicate': args['metadata']['colliders'][1]['path'] = '/World/Stool/seat'
    elif kind == 'stool_root': args['metadata']['colliders'][-1]['path'] = '/World/Stool'
    elif kind == 'missing_array': args['arrays'].pop('c000_f')
    elif kind == 'vertex_nan': args['arrays']['c000_v'][0, 0] = np.nan
    elif kind == 'face_outside': args['arrays']['c000_f'][0, 0] = 99
    elif kind == 'face_float': args['arrays']['c000_f'] = args['arrays']['c000_f'].astype(float)
    elif kind == 'face_degenerate': args['arrays']['c000_f'][0] = [0, 0, 1]
    elif kind == 'missing_cap': args['arrays']['c000_f'] = args['arrays']['c000_f'][1:]
    elif kind == 'overlapping_faces': args['arrays']['c000_f'][1] = args['arrays']['c000_f'][0]
    elif kind == 'noncircular_profile':
        args['arrays']['c000_v'][0, 0] += .01; update_bounds(args, 0)
    elif kind == 'rotated_world_axis':
        angle = .2; frame = np.eye(4)
        frame[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        args['metadata']['colliders'][0]['world_from_body'] = frame.tolist()
    elif kind == 'stale_bounds': args['metadata']['colliders'][0]['bounds_local'][0][0] -= .1
    elif kind == 'bad_frame': args['metadata']['colliders'][0]['world_from_body'][0][0] = 2.
    elif kind == 'reflection': args['metadata']['colliders'][0]['world_from_body'][0][0] = -1.
    else:
        index = 1 if kind.startswith('leg') else 5
        vertices = args['arrays'][f'c{index:03d}_v']
        if kind == 'leg_off_floor': vertices[vertices[:, 2] == 0., 2] = .01
        elif kind == 'leg_off_seat': vertices[vertices[:, 2] > .5, 2] -= .01
        elif kind == 'rung_below_floor': vertices[:, 2] -= .3
        else: vertices[:, 2] += .6
        update_bounds(args, index)
    with pytest.raises(ValueError): module().build_stool_geometry(**args)


@pytest.mark.parametrize('changes', [dict(new_seat_top_z_m=.02), dict(new_seat_top_z_m=.05),
    dict(new_seat_top_z_m=np.nan), dict(new_seat_top_z_m=True), dict(floor_z_m=np.inf),
    dict(stool_xy_translation_m=[0., np.inf]), dict(stool_xy_translation_m=[0., 0., 0.])])
def test_nonfinite_or_too_low_proposals_rejected(changes):
    args = fixture(); args.update(changes)
    with pytest.raises(ValueError): module().build_stool_geometry(**args)


@pytest.mark.parametrize('field', ['collision_geometry_sha256', 'collision_arrays_sha256', 'geometry_source_sha256'])
def test_unbound_source_hashes_rejected(field):
    args = fixture(); args['source_bindings'][field] = 'not-a-sha'
    with pytest.raises(ValueError): module().build_stool_geometry(**args)


def test_retired_geometry_and_native_claims_do_not_transfer():
    args = fixture(); args['source_bindings']['geometry_id'] = 'retired'
    with pytest.raises(ValueError): module().build_stool_geometry(**args)
    args = fixture(); args['metadata']['native_scene_installed'] = True
    with pytest.raises(ValueError): module().build_stool_geometry(**args)
