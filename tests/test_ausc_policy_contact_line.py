from types import SimpleNamespace
import numpy as np
import pytest

from sim.auscultation_expert_layout_v1.plan_contact_pose import candidate_geometry


def case():
    target = SimpleNamespace(position_world=np.array([.1, .2, .3]),
                             normal_world=np.array([0., -1., 0.]))
    normal = np.array([0., -np.cos(.3), np.sin(.3)])
    x = np.array([1., 0., 0.])
    rotation = np.column_stack([x, np.cross(-normal, x), -normal])
    candidate = dict(normal_world=normal, wrist_rotation_world=rotation, gap_m=.002)
    return target, candidate


def test_new_surface_fit_uses_the_same_line_as_the_normal_controller():
    target, candidate = case()
    candidate['gap_axis'] = 'command_normal'
    geometry = candidate_geometry(target, candidate, np.eye(4), pre_gap=.025)
    direction = candidate['normal_world']
    for key, gap in [('center_world', .002), ('pre_world', .025), ('free_end_world', .015)]:
        np.testing.assert_allclose(geometry[key], target.position_world+direction*gap, atol=1e-14)
    np.testing.assert_array_equal(target.normal_world, [0., -1., 0.])


def test_historical_default_is_preserved_but_unknown_gap_axis_is_rejected():
    target, candidate = case()
    geometry = candidate_geometry(target, candidate, np.eye(4))
    np.testing.assert_array_equal(geometry['center_world'], target.position_world+target.normal_world*.002)
    candidate['gap_axis'] = 'silently_changed'
    with pytest.raises(ValueError, match='gap axis'):
        candidate_geometry(target, candidate, np.eye(4))


def test_varied_patient_targets_are_bound_to_actual_installed_receipt(tmp_path):
    import hashlib,json
    from sim.ausc_policy.geometry_plan import patient_provenance
    target=dict(name='R3',position_world_m=[0.,1.,1.3],normal_world=[0.,-1.,0.])
    receipt=dict(targets=[target],non_patient_scene_unchanged=True,identity='p000875',
                 asset_sha256='a'*64,identity_split_sha256='b'*64)
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    config=dict(targets=[dict(target)],patient_receipt=dict(file=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert patient_provenance(config)['identity']=='p000875'
    config['targets'][0]['position_world_m']=[.005,1.,1.3]
    with pytest.raises(ValueError,match='targets differ'):
        patient_provenance(config)
    config['targets'][0]=dict(target);path.write_text(json.dumps(receipt)+' ')
    with pytest.raises(ValueError,match='receipt changed'):
        patient_provenance(config)
