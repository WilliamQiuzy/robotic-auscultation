"""Synthetic-only tests; no native patient geometry is read on the Mac."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy.contact_surface_gap import sha


@pytest.fixture
def audit_module():
    assert importlib.util.find_spec('sim.ausc_policy.contact_layer_audit') is not None, 'Layer audit is not implemented'
    return importlib.import_module('sim.ausc_policy.contact_layer_audit')


def plane(z=0.):
    return np.array([[[-1, -1, z], [1, -1, z], [1, 1, z]],
                     [[-1, -1, z], [1, 1, z], [-1, 1, z]]], float)


def test_first_hit_returns_original_triangle_index_and_signed_offset(audit_module):
    triangles = np.concatenate([plane(-.01), plane()])
    offsets, indices = audit_module.ray_hits([[.2, -.1, .003], [.2, -.1, -.002]],
                                            [0, 0, -1], triangles)
    assert offsets == pytest.approx([.003, -.002])
    assert indices.tolist() == [2, 2]


def test_first_hit_is_from_segment_start_not_nearest_absolute_offset(audit_module):
    offsets, indices = audit_module.ray_hits([[.2, -.1, 0]], [0, 0, -1],
                                            np.concatenate([plane(-.001), plane(.04)]))
    assert offsets == pytest.approx([-.04])
    assert indices.tolist() == [2]


def test_missing_parallel_empty_and_out_of_bounds_are_nan_minus_one(audit_module):
    for points, direction, mesh in [([[2, 2, 0]], [0, 0, -1], plane()),
                                    ([[0, 0, 0]], [1, 0, 0], plane()),
                                    ([[0, 0, .051]], [0, 0, -1], plane()),
                                    ([[0, 0, 0]], [0, 0, -1], np.empty((0, 3, 3)))]:
        offsets, indices = audit_module.ray_hits(points, direction, mesh)
        assert np.isnan(offsets).all()
        assert indices.tolist() == [-1]
    offsets, _ = audit_module.ray_hits([[0, 0, .051]], [0, 0, -1], plane(), half_length_m=.06)
    assert offsets == pytest.approx([.051])


@pytest.mark.parametrize('direction', [[0, 0, 0], [0, 0, 2], [0, 0, np.nan], [[0, 0, 1]]])
def test_invalid_direction_rejected(audit_module, direction):
    with pytest.raises(ValueError):
        audit_module.ray_hits([[0, 0, 0]], direction, plane())


@pytest.mark.parametrize('length', [0, -.1, np.inf, np.nan, True])
def test_invalid_segment_length_rejected(audit_module, length):
    with pytest.raises(ValueError):
        audit_module.ray_hits([[0, 0, 0]], [0, 0, -1], plane(), half_length_m=length)


def test_nonfinite_input_rejected(audit_module):
    with pytest.raises(ValueError):
        audit_module.ray_hits([[np.nan, 0, 0]], [0, 0, -1], plane())
    with pytest.raises(ValueError):
        audit_module.ray_hits([[0, 0, 0]], [0, 0, -1], plane(np.inf))


def test_circular_patch_is_deterministic_and_preserves_anchor(audit_module):
    center = np.array([.3, -.2, 1.])
    normal = np.array([1., 2., 3.]) / np.sqrt(14.)
    points = audit_module.circular_patch(center, normal)
    assert points.shape == (97, 3)
    assert np.array_equal(points, audit_module.circular_patch(center, normal))
    assert np.array_equal(points[0], center)
    delta = points[1:] - center
    assert delta @ normal == pytest.approx(np.zeros(96), abs=1e-14)
    assert np.linalg.norm(delta, axis=1) == pytest.approx(np.repeat([.0115, .021, .023], 32))


def test_physical_transform_rejects_scale_reflection_shear_and_bad_bottom(audit_module):
    for transform in [np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]),
                      np.eye(3), np.full((4, 4), np.nan)]:
        with pytest.raises(ValueError):
            audit_module.physical_transform(transform)
    transform = np.eye(4)
    transform[0, 1] = .01
    with pytest.raises(ValueError):
        audit_module.physical_transform(transform)
    transform = np.eye(4)
    transform[3, 0] = .01
    with pytest.raises(ValueError):
        audit_module.physical_transform(transform)


SUFFIXES = {'skin': 'trans__organic__malecaucasianskin',
            'shirt': 'opaque__fabric__scrubsshirt',
            'pants': 'opaque__fabric__scrubspants',
            'shoes': 'opaque__leather__sneakers'}


def mesh_fixture():
    entries, arrays = [], {}
    for i, (layer, suffix) in enumerate(SUFFIXES.items()):
        key = f'c{i:03d}'
        entries.append(dict(path='/World/Patient/body/' + suffix, key=key, body='world',
                            approximation='none', world_from_body=np.eye(4).tolist()))
        arrays[key + '_v'] = plane({'skin': 0, 'shirt': .003, 'pants': -2, 'shoes': -3}[layer]).reshape(-1, 3)
        arrays[key + '_f'] = np.arange(6).reshape(2, 3)
    return entries, arrays


def test_layers_use_exact_suffixes_and_rigid_world_transform(audit_module):
    entries, arrays = mesh_fixture()
    entries[0]['world_from_body'][2][3] = .002
    colliders = audit_module.patient_meshes(entries, arrays)
    assert [c['layer'] for c in colliders] == list(SUFFIXES)
    assert colliders[0]['triangles'][:, :, 2] == pytest.approx(np.full((2, 3), .002))
    entries[0]['path'] += '_fake'
    with pytest.raises(ValueError, match='Unclassified'):
        audit_module.patient_meshes(entries, arrays)


@pytest.mark.parametrize('field,value', [('body', 'dynamic'), ('approximation', 'convexHull')])
def test_dynamic_or_approximated_patient_geometry_rejected(audit_module, field, value):
    entries, arrays = mesh_fixture()
    entries[0][field] = value
    with pytest.raises(ValueError, match='static'):
        audit_module.patient_meshes(entries, arrays)


def test_invalid_faces_and_mesh_vertices_rejected(audit_module):
    entries, arrays = mesh_fixture()
    arrays['c000_f'][0, 0] = 50
    with pytest.raises(ValueError):
        audit_module.patient_meshes(entries, arrays)


def test_separate_layers_report_first_hit_and_normal_winding(audit_module):
    entries, arrays = mesh_fixture()
    arrays['c000_f'] = arrays['c000_f'][:, ::-1]
    colliders = audit_module.patient_meshes(entries, arrays)
    result = audit_module.layer_rays([[.2, -.1, .01], [2, 2, .01]], [0, 0, -1], colliders)
    assert result['layers']['skin']['hit_count'] == 1
    assert result['layers']['skin']['missing_count'] == 1
    assert result['layers']['skin']['signed_offset_m'] == pytest.approx([.01, None])
    assert result['layers']['skin']['geometry_normal_angle_deg'] == [180., None]
    assert result['layers']['skin']['unoriented_normal_angle_deg'] == [0., None]
    assert result['layers']['all_patient']['first_hit_layer'] == ['shirt', None]
    assert result['layers']['all_patient']['first_hit_collider'][0] == entries[1]['path']
    assert result['layers']['all_patient']['triangle_index'] == [0, -1]
    assert result['shirt_skin_separation_m'] == pytest.approx([.003, None])
    assert result['paired_skin_shirt_hit_count'] == 1
    assert result['missing_skin_coverage'] is True


def test_hole_and_empty_skin_do_not_become_zero_or_global_absence(audit_module):
    entries, arrays = mesh_fixture()
    arrays['c000_v'] += [5, 0, 0]
    result = audit_module.layer_rays([[0, 0, .01]], [0, 0, -1], audit_module.patient_meshes(entries, arrays))
    assert result['layers']['skin']['signed_offset_stats_m'] is None
    assert result['layers']['skin']['signed_offset_m'] == [None]
    assert result['shirt_skin_separation_m'] == [None]
    assert 'globally' in result['missing_interpretation']


def test_aabb_prefilter_preserves_native_triangle_row_identity(audit_module):
    entries, arrays = mesh_fixture()
    arrays['c000_v'][:3] += [5, 0, 0]
    result = audit_module.layer_rays([[-.2, .1, .01]], [0, 0, -1], audit_module.patient_meshes(entries, arrays))
    assert result['layers']['skin']['triangle_index'] == [1]


def test_topology_counts_unused_vertices_boundary_and_nonmanifold_edges(audit_module):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 5]], float)
    result = audit_module.mesh_topology(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
    assert result['unused_vertex_count'] == 1
    assert result['boundary_edge_count'] == 4
    assert result['nonmanifold_edge_count'] == 0
    assert result['surface_coverage_proven'] is False


def native_fixture(tmp_path):
    run = tmp_path / 'v10'
    case = run / 'output/review'
    geometry = case / 'geometry'
    replay = case / 'fresh_expert'
    geometry.mkdir(parents=True)
    replay.mkdir()
    entries, arrays = mesh_fixture()
    for i in range(32):
        angles = [i * np.pi / 16, (i + 1) * np.pi / 16]
        front = np.array([[r * np.cos(a), r * np.sin(a), -.047]
                          for r in (.021, .023) for a in angles], np.float32)
        key = f'r{i}'
        arrays[key + '_v'] = np.r_[front, front + [0, 0, .0015]]
        arrays[key + '_f'] = np.array([[0, 1, 2]])
        entries.append(dict(path=f'/World/Stethoscope/Carriage/DeviceContactRing/sector_{i:02d}',
                            body='/World/Stethoscope/Carriage', approximation='convexHull', key=key,
                            world_from_body=np.eye(4).tolist()))
    (geometry / 'collision_geometry.json').write_text(json.dumps(dict(colliders=entries)))
    np.savez(geometry / 'collision_geometry.npz', **arrays)
    artifacts = dict(collision_geometry_sha256=sha(geometry / 'collision_geometry.json'),
                     collision_arrays_sha256=sha(geometry / 'collision_geometry.npz'))
    names = [s + str(i) for i in range(1, 6) for s in 'LR']
    bindings = []
    for name in names:
        path = run / (name + '.json')
        path.write_text(json.dumps(dict(name=name, position_world=[0, 0, .003], normal_world=[0, 0, 1],
                                        geometry_artifacts=artifacts)))
        bindings.append(dict(name=name, file=str(path), sha256=sha(path)))
    rows = [dict(phase='HOLD', target=name, time_s=i / 240, axis_world=[0, 0, -1])
            for i, name in enumerate(np.repeat(names, 720))]
    trace = replay / 'fresh_expert_trace.jsonl'
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    indices = np.arange(7, len(rows), 8)
    poses = np.tile([0, 0, .055, 1, 0, 0, 0], (len(indices), 1, 1))
    np.savez(replay / 'fresh_expert_replay.npz', paths=np.array(['/World/Stethoscope/Carriage']),
             row_indices=indices, poses=poses)
    native = dict(geometry_artifacts=artifacts, plan_bindings=bindings, physics_steps=len(rows),
                  trace_sha256=sha(trace))
    (case / 'fresh_expert_result.json').write_text(json.dumps(native))
    return run, case, native, sha(replay / 'fresh_expert_replay.npz')


def test_synthetic_full_audit_binds_sources_middle_pose_order_and_false_flags(audit_module, tmp_path, monkeypatch):
    run, case, _, replay_sha = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path / 'unique-audit'
    report = audit_module.audit(run, output, expected_replay_sha256=replay_sha)
    assert json.loads((output / 'summary.json').read_text())['schema'] == report['schema']
    assert [p['target'] for p in report['points']] == [s + str(i) for i in range(1, 6) for s in 'LR']
    assert report['points'][0]['native_middle_hold']['row'] == 367
    assert report['points'][0]['anchor_patch']['ray_count'] == 97
    assert report['points'][0]['native_middle_hold']['circular_patch']['ray_count'] == 97
    assert report['points'][0]['native_middle_hold']['retainer_front']['ray_count'] == 96
    assert report['source_sha256_before'] == report['source_sha256_after']
    assert report['expected_replay_sha256'] == replay_sha
    assert all(value is False for value in report['qualification'].values())
    assert len(report['plan_bindings']) == 10
    with pytest.raises(FileExistsError):
        audit_module.audit(run, output, expected_replay_sha256=replay_sha)


@pytest.mark.parametrize('corruption', ['plan', 'geometry', 'trace', 'replay', 'order'])
def test_source_binding_and_ledger_rejection(audit_module, tmp_path, monkeypatch, corruption):
    run, case, native, replay_sha = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    if corruption == 'plan':
        Path(native['plan_bindings'][0]['file']).write_text('{}')
    elif corruption == 'geometry':
        (case / 'geometry/collision_geometry.json').write_text('{}')
    elif corruption == 'trace':
        with (case / 'fresh_expert/fresh_expert_trace.jsonl').open('a') as f:
            f.write('{}\n')
    elif corruption == 'replay':
        np.savez(case / 'fresh_expert/fresh_expert_replay.npz', paths=np.array(['/bad']), row_indices=[7], poses=np.zeros((1, 1, 7)))
    else:
        native['plan_bindings'].reverse()
        (case / 'fresh_expert_result.json').write_text(json.dumps(native))
    with pytest.raises(ValueError):
        audit_module.audit(run, tmp_path / 'rejected', expected_replay_sha256=replay_sha)
    assert not (tmp_path / 'rejected').exists()


def test_raw_output_requires_linux_cpu_and_cannot_modify_source_run(audit_module, tmp_path, monkeypatch):
    run, _, _, replay_sha = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'darwin')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='Linux CPU'):
        audit_module.audit(run, tmp_path / 'mac-output', expected_replay_sha256=replay_sha)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='Linux CPU'):
        audit_module.audit(run, tmp_path / 'gpu-output', expected_replay_sha256=replay_sha)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='outside'):
        audit_module.audit(run, run / 'new-output', expected_replay_sha256=replay_sha)


def test_sources_changed_during_audit_rejected_without_output(audit_module, tmp_path, monkeypatch):
    run, _, native, replay_sha = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    original = audit_module.layer_rays

    def change_plan(*args, **kwargs):
        Path(native['plan_bindings'][0]['file']).write_text('{}')
        return original(*args, **kwargs)

    monkeypatch.setattr(audit_module, 'layer_rays', change_plan)
    with pytest.raises(ValueError, match='Source files changed'):
        audit_module.audit(run, tmp_path / 'mutated-output', expected_replay_sha256=replay_sha)
    assert not (tmp_path / 'mutated-output').exists()


def test_split_hold_sequence_rejected_even_with_rebound_trace_hash(audit_module, tmp_path, monkeypatch):
    run, case, native, replay_sha = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    trace = case / 'fresh_expert/fresh_expert_trace.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[15]['target'], rows[727]['target'] = rows[727]['target'], rows[15]['target']
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    native['trace_sha256'] = sha(trace)
    (case / 'fresh_expert_result.json').write_text(json.dumps(native))
    with pytest.raises(ValueError, match='HOLD'):
        audit_module.audit(run, tmp_path / 'split-output', expected_replay_sha256=replay_sha)


def test_replay_expectation_is_required(audit_module, tmp_path, monkeypatch):
    run, _, _, _ = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(TypeError, match='expected_replay_sha256'):
        audit_module.audit(run, tmp_path / 'no-expectation')


@pytest.mark.parametrize('expected', ['', None, '0' * 63, 'z' * 64, '0' * 64])
def test_invalid_or_wrong_replay_expectation_rejected(audit_module, tmp_path, monkeypatch, expected):
    run, _, _, _ = native_fixture(tmp_path)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='replay SHA'):
        audit_module.audit(run, tmp_path / 'invalid-expectation', expected_replay_sha256=expected)
    assert not (tmp_path / 'invalid-expectation').exists()


def test_well_shaped_translated_replay_rejected_before_np_load(audit_module, tmp_path, monkeypatch):
    run, case, _, replay_sha = native_fixture(tmp_path)
    path = case / 'fresh_expert/fresh_expert_replay.npz'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays['poses'][:, :, 0] += .01
    np.savez(path, **arrays)
    monkeypatch.setattr(audit_module.sys, 'platform', 'linux')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    def forbidden_load(*args, **kwargs):
        pytest.fail('Changed replay must be rejected before any native arrays are loaded')

    monkeypatch.setattr(audit_module.np, 'load', forbidden_load)
    with pytest.raises(ValueError, match='replay SHA'):
        audit_module.audit(run, tmp_path / 'translated-replay', expected_replay_sha256=replay_sha)
    assert not (tmp_path / 'translated-replay').exists()
