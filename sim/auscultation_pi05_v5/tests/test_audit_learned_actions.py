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


DT = 1 / 240
CHECKPOINT = 'a' * 64


def auditor():
    name = 'sim.auscultation_pi05_v5.audit_learned_actions'
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


def fixture(tmp_path, *, new_target=False, unexplained_gap=False, reset_between=False):
    sensor_root = tmp_path / 'sensors'
    sensor_root.mkdir()
    binding = dict(checkpoint_metadata_sha256=CHECKPOINT, data_hash='b' * 64,
                   inference_source_sha256={'actor.py': 'c' * 64}, sampling_steps=10,
                   sampling_seed=917, sampling_policy='fixed', audio='absent', modalities={'rgb': 'camera'},
                   action_semantics='measured_joint_command_offset_v1',
                   training_recipe='phase_balanced_measured_joint_actions_v5')
    physics, frames, learned, requests = [], [], [], []

    def append(command, target='R1', subphase='settle'):
        measured = np.asarray(np.asarray(command) + .001, dtype=np.float32)
        physics.append(dict(time_s=(len(physics) + 1) * DT, q_command=list(command),
                            q=measured.tolist(), target=target, phase='APPROACH', subphase=subphase, recovery=False))

    for _ in range(3):
        append(np.zeros(7))
    otg, inp, out = generator(np.zeros(7))
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
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + number
        raw_path = sensor_root / ('%05d.npz' % number)
        np.savez_compressed(raw_path, rgb=rgb, depth_m=np.ones((3, 4)), depth_valid=np.ones((3, 4), bool))
        raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        frame = dict(index=number, physics_row=start-1, time_s=physics[-1]['time_s'], phase=phase,
                     target=target, q=measured.tolist(), state=state.tolist(), raw_file=raw_path.name,
                     raw_sha256=raw_sha, target_estimate={'revision': 0})
        frames.append(frame)
        raw = np.zeros((10, 7))
        if number == 0:
            distance = .005 if new_target else .01
            raw[0] = np.array([distance] * 6 + [-distance]) - measured
        else:
            raw[0] = np.array([-.003] * 6 + [.003])
        request_id = 'request-' + str(number)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format='PNG')
        request_bytes = json.dumps(dict(request_id=request_id, phase=phase, target=target,
                                       state=state.tolist(), rgb_png=base64.b64encode(buf.getvalue()).decode())).encode()
        hashes = dict(request_payload_sha256=hashlib.sha256(request_bytes).hexdigest(),
                      rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb_shape=list(rgb.shape),
                      state_f32_sha256=f32_digest(state), raw_action_f32_sha256=f32_digest(raw))
        receipt = dict(binding, request_id=request_id, raw_actions=raw.tolist(), **hashes)
        requests.append(dict(request_id=request_id, phase=phase, target=target,
                             checkpoint_metadata_sha256=CHECKPOINT, **hashes))
        goal = measured + raw[0]
        inp.target_position = goal.tolist()
        for _ in range(12):
            assert otg.update(inp, out) in (ruckig.Result.Working, ruckig.Result.Finished)
            append(out.new_position, target=target, subphase='learned_' + phase.lower())
            out.pass_to_input(inp)
        learned.append(dict(index=number, phase=phase, target=target, time_s=frame['time_s'],
                            observation_frame=number, raw_sensor_sha256=raw_sha,
                            physics_row_start=start, physics_row_end_exclusive=len(physics),
                            accepted=True, expert_fallback=False, executed_chunk_prefix=1,
                            raw_actions=raw.tolist(), measured_q=measured.tolist(), goal_q=goal.tolist(), inference=receipt))
    return dict(learned=learned, frames=frames, physics=physics, binding=binding,
                expected_checkpoint_metadata_sha256=CHECKPOINT, raw_sensor_root=sensor_root,
                server_requests=requests, velocity=.25, task_scope='full')


def test_replays_continuous_commands_and_independently_known_first_tick(tmp_path):
    data = fixture(tmp_path)
    # For the first short tick, |a| < .5, |v| < .25: q = jerk*t^3/6.
    expected = np.array([1.] * 6 + [-1.]) * 8 * DT**3 / 6
    np.testing.assert_allclose(data['physics'][3]['q_command'], expected, atol=1e-14, rtol=0)
    report = auditor().audit_records(**data)
    assert report['passed'], report
    assert report['verified_executed_steps'] == 24
    assert report['sensor_payload_binding_verified']
    assert report['server_request_log_verified']
    assert report['policy_acceptance'] == 'not_evaluated'


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


def test_file_audit_binds_runtime_execution_snapshot(tmp_path):
    data = fixture(tmp_path)
    module = auditor()
    for name, key in [('learned_actions.jsonl', 'learned'), ('expert_trace.jsonl', 'physics'),
                      ('sensors/frames.jsonl', 'frames'), ('server_requests.jsonl', 'server_requests')]:
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in data[key]))
    execution = importlib.import_module('sim.auscultation_pi05_v5.execution')
    runtime_bytes = Path(execution.__file__).read_bytes()
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    snapshot = tmp_path / 'input_snapshot'
    (snapshot / 'objects').mkdir(parents=True)
    (snapshot / 'objects' / runtime_sha).write_bytes(runtime_bytes)
    snapshot_manifest = snapshot / 'manifest.json'
    snapshot_manifest.write_text(json.dumps(dict(files={'source/pi05_v5/execution.py': {'sha256': runtime_sha}})))
    outcome = dict(learned_policy_binding=data['binding'], task_scope='full', success=False,
                   learned_command_limits=dict(joint_velocity_rad_s=.25, joint_acceleration_rad_s2=.5, joint_jerk_rad_s3=8.),
                   input_snapshot='input_snapshot/manifest.json',
                   input_snapshot_manifest_sha256=hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest())
    (tmp_path / 'outcome.json').write_text(json.dumps(outcome))
    report = module.audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT,
                                  server_requests_path=tmp_path / 'server_requests.jsonl')
    assert report['passed'], report
    assert report['runtime_execution_snapshot_verified']
    (snapshot / 'objects' / runtime_sha).write_bytes(b'altered runtime')
    with pytest.raises(ValueError, match='snapshot'):
        module.audit_episode(tmp_path, expected_checkpoint_metadata_sha256=CHECKPOINT)
