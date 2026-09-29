"""Independent synthetic evidence with real Ruckig and an analytic first tick."""
import base64
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import ruckig
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_vla_v2.hall import HallPacket, default_calibration
from sim.auscultation_pi05_v6.task_observation import observation_binding
from sim.auscultation_pi05_v6.action_encoding import action_encoding_binding


DT = 1 / 240
CHECKPOINT = 'a' * 64
SEED = np.array([.2, -.4, .3, 1.2, -.2, .1, .4])


def auditor():
    name = 'sim.auscultation_pi05_v6.audit_learned_actions'
    assert importlib.util.find_spec(name) is not None, 'saved learned-action auditor required'
    return importlib.import_module(name)


def f32_digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


def generator(q):
    otg, inp, out = ruckig.Ruckig(7, DT), ruckig.InputParameter(7), ruckig.OutputParameter(7)
    inp.current_position = list(q)
    inp.current_velocity = inp.current_acceleration = [0.] * 7
    inp.target_velocity = inp.target_acceleration = [0.] * 7
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [.25] * 7, [.5] * 7, [8.] * 7
    return otg, inp, out


def fixture(tmp_path, *, new_target=False, unexplained_gap=False, reset_between=False, add_measured_origin=False):
    sensor_root = tmp_path / 'sensors'
    sensor_root.mkdir()
    binding = dict(checkpoint_metadata_sha256=CHECKPOINT, data_hash='b' * 64,
                   inference_source_sha256={'actor.py': 'c' * 64}, sampling_steps=10,
                   sampling_seed=917, sampling_policy='fixed', audio='absent', modalities={'rgb': 'camera'},
                   dynamics_recipe_sha256='d'*64, dynamics_contract_sha256='e'*64,
                   execution_contract_sha256='f'*64,
                   action_semantics='absolute_joint_command_v1',
                   training_recipe='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1',
                   observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding())
    physics, frames, learned, requests = [], [], [], []
    body_poses=[];kin=OpenArmKinematics();cal=default_calibration()
    registrations={}
    for target,x in [('R1',.1),('R2',.2)]:
        task=np.eye(4);task[:3,3]=[x,.2,.4]
        registrations[target]=dict(initial_task_position=task[:3,3].tolist(),
            initial_task_rotation=task[:3,:3].tolist(),task_pose_in_body_reference=task.tolist(),
            initial_body_reference_to_current_base=np.eye(4).tolist(),source_frame=0,
            source_time_s=3*DT,registration_observed_target='R1')

    def append(command, target='R1', subphase='settle'):
        measured = np.asarray(np.asarray(command) + .001, dtype=np.float32)
        physics.append(dict(time_s=(len(physics) + 1) * DT, q_command=list(command),
                            q=measured.tolist(), target=target, phase='APPROACH', subphase=subphase, recovery=False))

    for _ in range(3):
        append(SEED)
    otg, inp, out = generator(SEED)
    for number in range(2):
        target = 'R2' if number and new_target else 'R1'
        if number and new_target:
            inp.target_position = physics[-1]['q_command']
            for _ in range(120):
                assert otg.update(inp, out) in (ruckig.Result.Working, ruckig.Result.Finished)
                append(out.new_position, subphase='learned_release_stop')
                out.pass_to_input(inp)
                if max(abs(x) for x in out.new_velocity + out.new_acceleration) < 1e-10:
                    break
            else:
                raise AssertionError('fixture release did not stop')
            for _ in range(3):
                append(physics[-1]['q_command'], subphase='withdraw_settle')
            otg, inp, out = generator(physics[-1]['q_command'])
        elif number and unexplained_gap:
            append(physics[-1]['q_command'], subphase='unlogged_route')
        if number and reset_between:
            otg, inp, out = generator(physics[-1]['q_command'])
        start = len(physics)
        measured = np.asarray(physics[-1]['q'])
        phase = 'RETRACT' if new_target and number == 0 else 'FREE'
        state = np.zeros(32, dtype=np.float32)
        state[:7] = measured
        packet=HallPacket(27000,physics[-1]['time_s'],physics[-1]['time_s'],cal.sha256,'simulated_adc')
        decoded=cal.decode(packet,now_s=physics[-1]['time_s'])
        state[20:22]=[decoded.hall_voltage_v,decoded.spring_force_estimate_n]
        body=np.eye(4);body[0,3]=number*.001
        task=body@np.asarray(registrations[target]['task_pose_in_body_reference'])
        tcp=kin.tcp(measured,decoded.displacement_estimate_m)
        relative=tcp[:3,:3].T@task[:3,:3]
        policy_state=np.r_[state,tcp[:3,:3].T@(task[:3,3]-tcp[:3,3]),relative[:,:2].T.reshape(-1)].astype(np.float32)
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + number
        raw_path = sensor_root / ('%05d.npz' % number)
        np.savez_compressed(raw_path, rgb=rgb, depth_m=np.ones((3, 4)), depth_valid=np.ones((3, 4), bool))
        raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        frame = dict(index=number, physics_row=start-1, time_s=physics[-1]['time_s'], phase=phase,
                     target=target, q=measured.tolist(), state=state.tolist(), raw_file=raw_path.name,
                     raw_sha256=raw_sha, target_estimate={'revision': 0},hall=vars(packet))
        frames.append(frame)
        body_poses.append(dict(frame_index=number,time_s=frame['time_s'],
            body_reference_to_current_base=body.tolist(),source='measured RGB-D rigid registration'))
        raw = np.zeros((10, 7))
        if number == 0:
            distance = .005 if new_target else .01
            raw[0] = SEED + np.array([distance] * 6 + [-distance])
        else:
            raw[0] = SEED + np.array([-.003] * 6 + [.003])
        request_id = 'request-' + str(number)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format='PNG')
        request_bytes = json.dumps(dict(request_id=request_id, phase=phase, target=target,
                                       state=policy_state.tolist(), rgb_png=base64.b64encode(buf.getvalue()).decode())).encode()
        hashes = dict(request_payload_sha256=hashlib.sha256(request_bytes).hexdigest(),
                      rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb_shape=list(rgb.shape),
                      state_f32_sha256=f32_digest(policy_state), raw_action_f32_sha256=f32_digest(raw))
        receipt = dict(binding, request_id=request_id, raw_actions=raw.tolist(), **hashes)
        requests.append(dict(request_id=request_id, phase=phase, target=target,
                             checkpoint_metadata_sha256=CHECKPOINT, **hashes))
        goal = raw[0] + measured if add_measured_origin else raw[0].copy()
        inp.target_position = goal.tolist()
        for _ in range(12):
            assert otg.update(inp, out) in (ruckig.Result.Working, ruckig.Result.Finished)
            append(out.new_position, target=target, subphase='learned_' + phase.lower())
            out.pass_to_input(inp)
        learned.append(dict(index=number, phase=phase, target=target, time_s=frame['time_s'],
                            observation_frame=number, raw_sensor_sha256=raw_sha,
                            policy_state=policy_state.tolist(),observation_contract=observation_binding(),
                            physics_row_start=start, physics_row_end_exclusive=len(physics),
                            accepted=True, expert_fallback=False, executed_chunk_prefix=1,
                            raw_actions=raw.tolist(), measured_q=measured.tolist(), goal_q=goal.tolist(), inference=receipt))
    return dict(learned=learned, frames=frames, physics=physics, binding=binding,
                expected_checkpoint_metadata_sha256=CHECKPOINT, raw_sensor_root=sensor_root,
                server_requests=requests, velocity=.25, task_scope='full',body_poses=body_poses,
                registrations=registrations,kinematics=kin,hall_calibration=cal)


def test_replays_continuous_commands_and_independently_known_first_tick(tmp_path):
    data = fixture(tmp_path)
    # For the first short tick, |a| < .5, |v| < .25: q = jerk*t^3/6.
    expected = SEED + np.array([1.] * 6 + [-1.]) * 8 * DT**3 / 6
    np.testing.assert_allclose(data['physics'][3]['q_command'], expected, atol=1e-14, rtol=0)
    report = auditor().audit_records(**data)
    assert report['passed'], report
    assert report['verified_executed_steps'] == 24
    assert report['sensor_payload_binding_verified']
    assert report['server_request_log_verified']
    assert report['policy_acceptance'] == 'not_evaluated'
    assert report['policy_task_observation_verified']
    assert np.max(np.abs(data['learned'][0]['measured_q'])) > 1.
    np.testing.assert_array_equal(data['learned'][0]['goal_q'], data['learned'][0]['raw_actions'][0])


def test_rejects_offset_composition_even_if_all_recorded_commands_match_it(tmp_path):
    data = fixture(tmp_path, add_measured_origin=True)
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert 'absolute joint goal' in ' '.join(report['errors']).lower(), report


@pytest.mark.parametrize('field,value', [
    ('action_semantics', 'measured_joint_command_offset_v1'),
    ('training_recipe', 'phase_balanced_measured_joint_actions_v5'),
])
def test_rejects_legacy_semantics_even_when_every_receipt_agrees(tmp_path, field, value):
    data = fixture(tmp_path)
    data['binding'][field] = value
    for row in data['learned']:
        row['inference'][field] = value
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert 'policy binding' in ' '.join(report['errors']).lower(), report


@pytest.mark.parametrize('joint,value', [(1, .2), (3, -.001), (6, 1.6)])
def test_absolute_urdf_limit_violation_rejected_before_replay(tmp_path, joint, value):
    data = fixture(tmp_path)
    row = data['learned'][0]
    row['raw_actions'][0][joint] = value
    row['inference']['raw_actions'][0][joint] = value
    row['goal_q'][joint] = value
    action_hash = f32_digest(row['raw_actions'])
    row['inference']['raw_action_f32_sha256'] = action_hash
    data['server_requests'][0]['raw_action_f32_sha256'] = action_hash
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert 'URDF bounds' in ' '.join(report['errors']), report


@pytest.mark.parametrize('mutation,match', [
    ('goal', 'goal'), ('command', 'Ruckig'), ('origin', 'measured'),
    ('checkpoint', 'checkpoint'), ('raw_seventh_joint', 'raw'),
    ('state_hash', 'state_f32'), ('request_id', 'request'), ('frame', 'frame'),
])
def test_rejects_tampered_independent_evidence(tmp_path, mutation, match):
    data = fixture(tmp_path)
    record = data['learned'][0]
    if mutation == 'goal': record['goal_q'][0] += .001
    elif mutation == 'command': data['physics'][3]['q_command'][0] += .00001
    elif mutation == 'origin': record['measured_q'][0] += .001
    elif mutation == 'checkpoint': record['inference']['checkpoint_metadata_sha256'] = 'd' * 64
    elif mutation == 'raw_seventh_joint': record['raw_actions'][0][6] = 0.
    elif mutation == 'state_hash': record['inference']['state_f32_sha256'] = 'd' * 64
    elif mutation == 'request_id': record['inference']['request_id'] = 'unlogged-request'
    elif mutation == 'frame': record['observation_frame'] = 1
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert match.lower() in ' '.join(report['errors']).lower(), report


def test_rejects_reset_between_same_target_queries(tmp_path):
    report = auditor().audit_records(**fixture(tmp_path, reset_between=True))
    assert not report['passed'] and 'Ruckig' in ' '.join(report['errors'])


def test_rejects_unexplained_gap(tmp_path):
    report = auditor().audit_records(**fixture(tmp_path, unexplained_gap=True))
    assert not report['passed'] and 'gap' in ' '.join(report['errors']).lower()


def test_replays_release_then_resets_at_new_target(tmp_path):
    report = auditor().audit_records(**fixture(tmp_path, new_target=True))
    assert report['passed'], report
    assert report['verified_release_stop_steps'] > 0
    assert report['trajectory_initializations'] == 2


@pytest.mark.parametrize('count', [0, 5])
def test_failure_retains_zero_or_partial_executed_prefix(tmp_path, count):
    data = fixture(tmp_path)
    last = data['learned'][-1]
    end = last['physics_row_start'] + count
    data['physics'] = data['physics'][:end]
    last.update(physics_row_end_exclusive=end, failure_reason='explicit safety gate rejected command')
    report = auditor().audit_records(**data)
    assert report['passed'], report
    assert report['verified_executed_steps'] == 12 + count
    assert report['failed_queries'] == 1
    assert report['policy_acceptance'] == 'not_evaluated'


def test_missing_raw_sensor_is_reported_unavailable(tmp_path):
    data = fixture(tmp_path)
    data['raw_sensor_root'] = None
    report = auditor().audit_records(**data)
    assert report['command_binding_verified']
    assert not report['sensor_payload_binding_verified']
    assert not report['passed']
    assert 'unavailable' in ' '.join(report['limitations']).lower()


def test_more_than_twelve_ticks_is_rejected(tmp_path):
    data = fixture(tmp_path)
    data['learned'][0]['physics_row_end_exclusive'] += 1
    report = auditor().audit_records(**data)
    assert not report['passed'] and '12' in ' '.join(report['errors'])


@pytest.mark.parametrize('scope,expected', [('contact', True), ('full', False)])
def test_classical_lower_approach_is_only_contact_component_preparation(tmp_path, scope, expected):
    data = fixture(tmp_path)
    data['task_scope'] = scope
    data['physics'][0]['subphase'] = 'lower_approach'
    data['physics'][1]['subphase'] = data['physics'][2]['subphase'] = 'align'
    report = auditor().audit_records(**data)
    assert report['passed'] is expected, report


def test_rejected_first_proposal_has_no_logged_composition_or_learned_ticks(tmp_path):
    data = fixture(tmp_path)
    data['learned'] = data['learned'][:1]
    data['frames'] = data['frames'][:1]
    data['body_poses'] = data['body_poses'][:1]
    record = data['learned'][0]
    record.update(accepted=False, failure_reason='ValueError: Joint position limit',
                  physics_row_end_exclusive=record['physics_row_start'])
    record.pop('measured_q')
    record.pop('goal_q')
    data['physics'] = data['physics'][:record['physics_row_start']]
    report = auditor().audit_records(**data)
    assert report['evidence_consistent'], report
    assert not report['passed']
    assert report['verified_executed_steps'] == 0
    assert report['sensor_payload_binding_verified']


def write_episode(tmp_path, data):
    for name, key in [('learned_actions.jsonl', 'learned'), ('expert_trace.jsonl', 'physics'),
                      ('sensors/frames.jsonl', 'frames'), ('server_requests.jsonl', 'server_requests'),
                      ('sensors/body_poses.jsonl','body_poses')]:
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in data[key]))
    execution = importlib.import_module('sim.auscultation_pi05_v6.execution')
    runtime_bytes = Path(execution.__file__).read_bytes()
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    snapshot = tmp_path / 'input_snapshot'
    (snapshot / 'objects').mkdir(parents=True)
    (snapshot / 'objects' / runtime_sha).write_bytes(runtime_bytes)
    snapshot_manifest = snapshot / 'manifest.json'
    action_contract = importlib.import_module('sim.auscultation_pi05_v6.action_contract')
    action_bytes = Path(action_contract.__file__).read_bytes()
    action_sha = hashlib.sha256(action_bytes).hexdigest()
    (snapshot / 'objects' / action_sha).write_bytes(action_bytes)
    files={'source/pi05_v6/execution.py': {'sha256': runtime_sha},
           'source/pi05_v6/action_contract.py': {'sha256': action_sha}}
    def bind(key,content):
        sha=hashlib.sha256(content).hexdigest();(snapshot/'objects'/sha).write_bytes(content)
        files[key]={'sha256':sha}
    for key,module in [('source/pi05_v6/task_observation.py','sim.auscultation_pi05_v6.task_observation'),
        ('source/code/kinematics.py','sim.auscultation_expert_layout_v1.kinematics'),
        ('source/code/tool_frames.py','sim.auscultation_expert_layout_v1.tool_frames'),
        ('source/vla/hall.py','sim.auscultation_vla_v2.hall'),
        ('source/vla/force_sensor.py','auscultation.mechanical_sensor.force_feedback.force_sensor')]:
        bind(key,Path(importlib.import_module(module).__file__).read_bytes())
    bind('source/urdf',data['kinematics'].urdf_path.read_bytes())
    calibration=Path(importlib.import_module('sim.auscultation_vla_v2.hall').__file__).with_name('calibration')/'veo6_tared_snapshot.json'
    bind('source/vla/hall_calibration.json',calibration.read_bytes())
    for target,registration in data['registrations'].items():
        bind(target+'/plan',json.dumps(dict(world_from_left_base=np.eye(4).tolist(),points=[dict(name=target,
            position_world=registration['initial_task_position'],command_rotation_world=registration['initial_task_rotation'])])).encode())
    snapshot_manifest.write_text(json.dumps(dict(files=files)))
    (tmp_path/'task_frame_registration.json').write_text(json.dumps(data['registrations']))
    outcome = dict(learned_policy_binding=data['binding'], task_scope='full', success=False,
                   task_frame_registration=data['registrations'],observation_contract=observation_binding(),
                   learned_command_limits=dict(joint_velocity_rad_s=.25, joint_acceleration_rad_s2=.5, joint_jerk_rad_s3=8.),
                   input_snapshot='input_snapshot/manifest.json',
                   input_snapshot_manifest_sha256=hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest())
    (tmp_path / 'outcome.json').write_text(json.dumps(outcome))
    return runtime_sha


def test_file_audit_binds_runtime_execution_snapshot(tmp_path):
    data = fixture(tmp_path)
    runtime_sha = write_episode(tmp_path, data)
    module = auditor()
    report = module.audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT,
                                  server_requests_path=tmp_path / 'server_requests.jsonl')
    assert report['passed'], report
    assert report['runtime_execution_snapshot_verified']
    assert report['runtime_observation_snapshot_verified']
    (tmp_path / 'input_snapshot/objects' / runtime_sha).write_bytes(b'altered runtime')
    with pytest.raises(ValueError, match='snapshot'):
        module.audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT)


def test_explicit_outcome_old_semantics_cannot_be_overridden_by_external_binding(tmp_path):
    data = fixture(tmp_path)
    write_episode(tmp_path, data)
    binding_path = tmp_path / 'external_binding.json'
    binding_path.write_text(json.dumps(data['binding']))
    outcome_path = tmp_path / 'outcome.json'
    outcome = json.loads(outcome_path.read_text())
    outcome['learned_policy_binding']['action_semantics'] = 'measured_joint_command_offset_v1'
    outcome_path.write_text(json.dumps(outcome))
    with pytest.raises(ValueError, match='policy binding'):
        auditor().audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT,
                                binding_path=binding_path)


def test_external_dynamics_binding_cannot_replace_outcome_binding(tmp_path):
    data = fixture(tmp_path)
    write_episode(tmp_path, data)
    data['binding']['dynamics_recipe_sha256'] = '1'*64
    binding_path = tmp_path / 'external_binding.json'
    binding_path.write_text(json.dumps(data['binding']))
    with pytest.raises(ValueError, match='differs from saved outcome'):
        auditor().audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT,
                                binding_path=binding_path)


def test_file_audit_rejects_changed_action_contract_snapshot(tmp_path):
    data = fixture(tmp_path)
    write_episode(tmp_path, data)
    snapshot = tmp_path / 'input_snapshot'
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    sha = manifest['files']['source/pi05_v6/action_contract.py']['sha256']
    (snapshot / 'objects' / sha).write_bytes(b'changed action contract')
    with pytest.raises(ValueError, match='snapshot: action_contract.py'):
        auditor().audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT)


@pytest.mark.parametrize('field', ['dynamics_recipe_sha256', 'dynamics_contract_sha256',
                                  'execution_contract_sha256'])
def test_dynamics_binding_is_required_and_pinned_per_reply(tmp_path, field):
    data = fixture(tmp_path)
    data['learned'][0]['inference'][field] = '1'*64
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert field in ' '.join(report['errors'])
    del data['binding'][field]
    report = auditor().audit_records(**data)
    assert not report['passed']
    assert 'Incomplete policy binding' in ' '.join(report['errors'])


def test_policy_state_tamper_rejected_even_after_request_hashes_are_rebound(tmp_path):
    data=fixture(tmp_path);row=data['learned'][0]
    row['policy_state'][32]+=.02
    with np.load(data['raw_sensor_root']/data['frames'][0]['raw_file']) as a:rgb=a['rgb']
    changed=dict(state_f32_sha256=f32_digest(row['policy_state']),request_payload_sha256=auditor().request_digest(
        rgb,row['policy_state'],row['phase'],row['target'],row['inference']['request_id']))
    row['inference'].update(changed);data['server_requests'][0].update(changed)
    report=auditor().audit_records(**data)
    assert not report['passed'] and 'policy task state' in ' '.join(report['errors']).lower(),report


def test_current_body_transform_must_be_same_acquired_frame(tmp_path):
    data=fixture(tmp_path);data['body_poses'][0]['time_s']+=DT
    report=auditor().audit_records(**data)
    assert not report['passed'] and 'body' in ' '.join(report['errors']).lower(),report


def test_query_observation_contract_is_required(tmp_path):
    data=fixture(tmp_path);del data['learned'][0]['observation_contract']
    report=auditor().audit_records(**data)
    assert not report['passed'] and 'observation contract' in ' '.join(report['errors']).lower(),report


def test_registration_cannot_be_replaced_with_consistently_rebound_new_goal(tmp_path):
    data=fixture(tmp_path);write_episode(tmp_path,data)
    path=tmp_path/'task_frame_registration.json';registration=json.loads(path.read_text())
    registration['R1']['initial_task_position'][0]+=.02
    registration['R1']['task_pose_in_body_reference'][0][3]+=.02
    path.write_text(json.dumps(registration))
    outcome_path=tmp_path/'outcome.json';outcome=json.loads(outcome_path.read_text())
    outcome['task_frame_registration']=registration;outcome_path.write_text(json.dumps(outcome))
    with pytest.raises(ValueError,match='Approved target'):
        auditor().audit_episode(tmp_path,expected_checkpoint_metadata_sha256=CHECKPOINT)


def test_file_audit_rejects_changed_hall_calibration_object(tmp_path):
    data=fixture(tmp_path);write_episode(tmp_path,data)
    snapshot=tmp_path/'input_snapshot';manifest=json.loads((snapshot/'manifest.json').read_text())
    identity=manifest['files']['source/vla/hall_calibration.json']['sha256']
    (snapshot/'objects'/identity).write_bytes(b'changed calibration')
    with pytest.raises(ValueError,match='snapshot'):
        auditor().audit_episode(tmp_path,expected_checkpoint_metadata_sha256=CHECKPOINT)
