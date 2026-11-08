"""Prescribed-pose control contracts only; no native qualification."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_ausc_policy_m2_contract import metadata as old_metadata


def module():
    from sim.ausc_policy import m2_given_contact_pose
    return m2_given_contact_pose


def bound(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inputs(tmp_path):
    point = dict(name='L1', position_world_m=[.2, .3, .8], normal_world=[0., 0., -1.])
    patient = dict(identity='p000001', split='train', asset_sha256='a'*64,
                   non_patient_scene_unchanged=True, targets=[point])
    receipt = bound(tmp_path, 'patient.json', patient)
    rotation = Rotation.from_euler('z', .7).as_matrix()
    plan = dict(old_metadata(), success=True, name='L1', position_world=point['position_world_m'],
        normal_world=point['normal_world'], command_rotation_world=rotation.tolist(),
        command_normal_world=[0., 0., -1.], world_from_left_base=np.eye(4).tolist(),
        patient_provenance=dict(identity=patient['identity'], asset_sha256='a'*64, binding=receipt))
    plan_ref = bound(tmp_path, 'plan.json', plan)
    return dict(point, manifest_ref=receipt), plan_ref, receipt


def reference(tmp_path):
    args = inputs(tmp_path)
    return module().build_given_reference(*args, np.eye(4), 0.)


def metadata(ref):
    return module().route_metadata(old_metadata(), ref)


def test_preserves_selected_full_roll_and_surface_origin(tmp_path):
    ref = reference(tmp_path)
    value = ref.snapshot()
    np.testing.assert_allclose(ref.pose_base[:3, 3], [.2, .3, .8])
    np.testing.assert_allclose(ref.pose_base[:3, :3], Rotation.from_euler('z', .7).as_matrix())
    np.testing.assert_allclose(ref.normal_base, [0, 0, -1])
    assert value['available_at_s'] == 0 and value['target_source'] == 'given_contact_pose'
    with pytest.raises(ValueError):
        ref.pose_base.setflags(write=True)
    value['reference_pose_base'][0][3] = 100
    assert ref.pose_base[0, 3] == .2


@pytest.mark.parametrize('fault', ['point', 'base', 'normal', 'stale_plan', 'stale_patient'])
def test_reference_rejects_disagreement_or_changed_sources(tmp_path, fault):
    point, plan, receipt = inputs(tmp_path)
    base = np.eye(4)
    if fault == 'point': point['position_world_m'] = [0, 0, 0]
    if fault == 'base': base[0, 3] = .1
    if fault == 'normal': point['normal_world'] = [0, 1, 0]
    if fault == 'stale_plan': open(plan['file'], 'a').write(' ')
    if fault == 'stale_patient': open(receipt['file'], 'a').write(' ')
    with pytest.raises(ValueError): module().build_given_reference(point, plan, receipt, base, 0.)


def test_provider_requires_premotion_binding_and_immutable_reference(tmp_path):
    ref = reference(tmp_path); meta = metadata(ref)
    provider = module().GivenPoseProvider(ref, 'episode', meta)
    state = dict(q=np.zeros(7), time_s=0.)
    controller = SimpleNamespace(env=SimpleNamespace(base=np.eye(4), rows=[], state=lambda: state),
                                  target=SimpleNamespace(name='L1'))
    provider.bind_before_motion(controller)
    first = provider(controller, state)
    assert first['confidence'] == 1 and first['revision'] == 0
    state['time_s'] = .05
    second = provider(controller, state)
    np.testing.assert_array_equal(first['estimated_target_base'], second['estimated_target_base'])
    meta['given_reference']['reference_pose_base'][0][3] = 999
    assert provider(controller, state)['estimated_target_base'][0, 3] == .2
    controller.env.base[0, 3] = .01
    with pytest.raises(ValueError, match='base'): provider(controller, state)


def test_late_reference_and_changed_binding_fail(tmp_path):
    ref = reference(tmp_path)
    provider = module().GivenPoseProvider(ref, 'episode', metadata(ref))
    c = SimpleNamespace(env=SimpleNamespace(base=np.eye(4), rows=[{}], state=lambda: dict(time_s=.05)))
    with pytest.raises(ValueError, match='before'): provider.bind_before_motion(c)
    args = inputs(tmp_path)
    late = module().build_given_reference(*args, np.eye(4), .1)
    c.env.rows = []; c.env.state = lambda: dict(time_s=0.)
    with pytest.raises(ValueError, match='available'): module().GivenPoseProvider(late, 'e', metadata(late)).bind_before_motion(c)
    open(ref.snapshot()['sources']['fresh_plan']['file'], 'a').write(' ')
    with pytest.raises(ValueError, match='hash'): ref.verify_sources()


def test_equal_dimensions_cannot_mix_noisy_and_given_routes(tmp_path):
    from sim.ausc_policy import m2_contract as contract
    ref = reference(tmp_path); meta = metadata(ref)
    contract.require_observation_contract(meta)
    state = contract.encode_state(np.zeros(7), ref.pose_base, 1., metadata=meta)
    assert state.shape == (17,)
    for bad in [dict(meta, task_id='old'), dict(meta, contract_id=contract.CONTRACT_ID),
                dict(meta, target_id='R3'), dict(meta, installed_patient_receipt_sha256='0'*64),
                dict(meta, observation_contract=contract.observation_contract()),
                dict(old_metadata(), target_source='given_contact_pose')]:
        with pytest.raises(ValueError): contract.require_metadata(bad)
    with pytest.raises(ValueError): contract.encode_state(np.zeros(7), np.eye(4), 1., metadata=meta)
    with pytest.raises(ValueError): contract.encode_state(np.zeros(7), ref.pose_base, .9, metadata=meta)
