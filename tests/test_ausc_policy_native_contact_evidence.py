"""Contact diagnostics must retain native semantics without changing verdicts."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    return importlib.import_module('sim.ausc_policy.native_contact_evidence')


def test_report_preserves_ordered_normal_faces_and_positive_separation():
    data = SimpleNamespace(position=[.022, 0., -.0473], normal=[0., 0., -1.],
        impulse=[0., 0., -.001], separation=.0003, face_index0=4294967295,
        face_index1=17)
    row = module().contact_point_record(data, ['retainer', 'patient'],
        actors=['carriage', 'static_patient'], callback_index=2)
    assert row['normal_world'] == [0., 0., -1.]
    assert row['face_index0'] == 4294967295 and row['face_index1'] == 17
    assert row['face_index0_available'] is False and row['face_index1_available'] is True
    assert row['colliders'] == ['retainer', 'patient']
    assert row['actors'] == ['carriage', 'static_patient']
    assert row['callback_index'] == 2 and not row['native_fields_missing']
    assert row['separation_m'] == .0003
    assert row['position_world'] == [.022, 0., -.0473]
    assert row['impulse_Ns'] == [0., 0., -.001]


def test_missing_native_fields_are_explicit_never_fabricated():
    data = SimpleNamespace(position=[1., 2., 3.], impulse=[0., 0., 1.], separation=0.)
    row = module().contact_point_record(data, ['a', 'b'])
    assert row['normal_world'] is None
    assert row['face_index0'] is None and row['face_index1'] is None
    assert set(row['native_fields_missing']) == {'normal', 'face_index0', 'face_index1'}
    assert row['actors'] is None and row['callback_index'] is None


def test_face_zero_is_valid_and_optional_binding_errors_preserve_original_point():
    class Data:
        position = [.022, 0., -.047]
        impulse = [0., 0., .001]
        separation = .0001
        face_index0 = 0
        @property
        def normal(self):
            raise RuntimeError('binding normal unavailable')
        @property
        def face_index1(self):
            raise RuntimeError('binding face unavailable')
    row = module().contact_point_record(Data(), ['a', 'b'])
    assert row['face_index0'] == 0 and row['face_index0_available'] is True
    assert row['face_index1_available'] is False
    assert row['normal_world'] is None and row['face_index1'] is None
    assert 'normal' in row['native_field_errors'] and 'face_index1' in row['native_field_errors']
    assert row['impulse_Ns'] == [0., 0., .001]


def test_header_diagnostic_errors_cannot_prevent_contact_ledger_recording():
    class Header:
        @property
        def actor0(self):
            raise RuntimeError('binding actor unavailable')
        @property
        def stage_id(self):
            raise RuntimeError('binding stage unavailable')
        type = 'CONTACT_PERSISTS'
    row = module().contact_header_record(Header(), str)
    assert row['actors'] is None and row['stage_id'] is None
    assert row['event_type'] == 'CONTACT_PERSISTS'
    assert set(row['header_field_errors']) == {'actors', 'stage_id'}


def test_native_pose_preserves_xyzw_rotation_and_actual_translation():
    raw = dict(ret_val=True, position=[.1, .2, .3],
               rotation=[0., 0., np.sqrt(.5), np.sqrt(.5)])
    row = module().native_pose_sample(lambda path: raw, '/carriage', 'callback')
    assert row['available'] and row['sample'] == 'callback'
    np.testing.assert_allclose(row['world_from_carriage'],
        [[0., -1., 0., .1], [1., 0., 0., .2], [0., 0., 1., .3], [0., 0., 0., 1.]], atol=1e-14)
    assert row['contact_generation_pose_claimed'] is False
    assert row['source'] == 'omni.physx.IPhysx.get_rigidbody_transformation'


@pytest.mark.parametrize('raw', [dict(ret_val=False),
    dict(ret_val=True, position=[0., 0., 0.], rotation=[0., 0., 0., 0.]),
    dict(ret_val=True, position=[0., float('nan'), 0.], rotation=[0., 0., 0., 1.]),
    dict(ret_val=True, position=[0., 0., 0.], rotation=[0., 0., 0., 2.])])
def test_invalid_native_pose_has_no_usd_or_fk_fallback(raw):
    row = module().native_pose_sample(lambda _: raw, '/carriage', 'pre_step')
    assert not row['available'] and row['world_from_carriage'] is None
    assert row['error'] and row['contact_generation_pose_claimed'] is False


def test_native_pose_api_error_is_visible_without_substituting_identity():
    def reader(_):
        raise RuntimeError('not available')
    row = module().native_pose_sample(reader, '/carriage', 'post_step')
    assert not row['available'] and 'not available' in row['error']
    assert row['world_from_carriage'] is None


def test_failure_snapshot_retains_same_tick_poses_and_callback_link():
    from test_ausc_policy_retainer_failure import native_failure
    from sim.ausc_policy.retainer_failure import latch_retainer_failure
    import json
    env, world = native_failure()
    env.contact_tick_evidence = dict(schema='native_contact_tick_v1', physics_row=0,
        pre_step={'available': False, 'world_from_carriage': None},
        callbacks=[{'callback_index': 0, 'sample': 'callback'}],
        post_step={'available': False, 'world_from_carriage': None})
    latch_retainer_failure(env, world)
    doc = json.loads(env._retainer_failure_payload)
    assert doc['contact_tick_evidence'] == env.contact_tick_evidence
    assert env.retainer_front_audit['passed'] is False


def test_first_positive_contact_is_saved_even_if_original_front_audit_passes(tmp_path):
    from test_ausc_policy_retainer_failure import native_failure
    import json
    env, world = native_failure(bad_force=0.)
    assert env.retainer_front_audit['passed']
    module().latch_first_positive_contact(env, world)
    payload = env._retainer_first_contact_payload
    env.rows.append({'time_s': 2/240})
    module().latch_first_positive_contact(env, world)
    assert payload == env._retainer_first_contact_payload
    doc = json.loads(payload)
    assert doc['physics_row'] == 0 and doc['event_kind'] == 'first_positive_retainer_contact'
    assert doc['physical_gates_unchanged'] and not doc['training_eligible']
    receipt = module().save_first_positive_contact(env, tmp_path)
    assert receipt['saved'] and (tmp_path/'retainer_first_contact.json').is_file()


def test_no_first_contact_artifact_when_all_impulses_zero():
    from test_ausc_policy_retainer_failure import native_failure
    env, world = native_failure()
    for point in env.contact_points:
        point['impulse_Ns'] = [0., 0., 0.]
    module().latch_first_positive_contact(env, world)
    assert not hasattr(env, '_retainer_first_contact_payload')


def test_executed_step_links_pre_callback_current_post_samples(monkeypatch):
    import sys
    from test_ausc_policy_retainer_failure import native_failure
    from sim.ausc_policy import fresh_expert as fresh
    fixture, world = native_failure()
    env = fresh.FreshContactEnv.__new__(fresh.FreshContactEnv)
    env.__dict__.update(fixture.__dict__)
    env.rows = []; env.t = 0.; env.slider = 7; env.passive_preload_N = 1.5
    env.robot = SimpleNamespace(apply_action=lambda _: None); env.video_recorder = None
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(world[:3, :3]).as_quat()
    env.bodies = [('/World/Stethoscope/Carriage', SimpleNamespace(
        get_world_pose=lambda: (world[:3, 3], quat[[3, 0, 1, 2]])))]
    samples = []
    def sample(self, label):
        samples.append(label)
        return dict(sample=label, available=True, world_from_carriage=world.tolist(),
                    contact_generation_pose_claimed=False)
    monkeypatch.setattr(fresh.FreshContactEnv, '_sample_native_carriage', sample, raising=False)
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types',
                        SimpleNamespace(ArticulationAction=SimpleNamespace))
    def step(self, *args):
        index = self.on_native_contact_report([SimpleNamespace(num_contact_data=3)])
        for point in self.contact_points:
            point['callback_index'] = index
        self.rows.append(dict(time_s=1/240, phase='CONTACT', target='R3', subphase='contact_cycle'))
        return dict(axis_world=[0., 0., -1.])
    monkeypatch.setattr(fresh.IsaacContactEnv, 'step', step)
    env.step(np.zeros(7), 'CONTACT', 'R3')
    assert samples == ['pre_step_current', 'callback_current', 'post_step_current']
    assert env.contact_tick_evidence['physics_row'] == 0
    assert env.contact_tick_evidence['callbacks'][0]['callback_index'] == 0
    assert not env.contact_tick_evidence['contact_generation_pose_available']
    assert env._retainer_failure_payload and env._retainer_first_contact_payload
    assert env.retainer_front_audit['passed'] is False


def test_first_contact_and_failure_share_compact_auxiliary_limit(tmp_path):
    from test_ausc_policy_retainer_failure import native_failure
    from test_ausc_policy_compact_expert_capture import make
    from sim.ausc_policy import fresh_expert as fresh
    from sim.ausc_policy.retainer_failure import latch_retainer_failure
    _, capture = make(tmp_path)
    capture.close('original failure')
    env, world = native_failure()
    module().latch_first_positive_contact(env, world)
    latch_retainer_failure(env, world)
    env.save = lambda _: None
    env.world = SimpleNamespace(stop=lambda: None)
    _, errors = fresh.finalize_probe(env, None, None, tmp_path, capture=capture)
    assert not errors
    assert env.retainer_first_contact_evidence['saved']
    assert env.retainer_failure_evidence['saved']
    for name in ('retainer_first_contact.json', 'retainer_failure.json'):
        assert (capture.output/name).is_file()
    used = sum(p.stat().st_size for p in capture.output.iterdir() if p.is_file())
    assert used + capture.AUXILIARY_RESERVE_BYTES <= capture.auxiliary_limit
