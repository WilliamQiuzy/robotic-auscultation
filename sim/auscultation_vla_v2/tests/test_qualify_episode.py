"""CPU replay/export tests. All data are synthetic tmp_path fixtures, not task data.

Successful integration fixtures stub ONLY the separate V1 mechanics/posture
auditor, whose real acceptance is covered elsewhere. RGB-D, tracker, Hall, FK,
state encoder, action labels, hashes and export execute normally here.
"""
from dataclasses import asdict
import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.auscultation_vla_v2.hall import HallSimulator, HallPacket, default_calibration
from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics, DEFAULT_URDF_PATH


def api():
    try: module = importlib.import_module('sim.auscultation_vla_v2.qualify_episode')
    except ModuleNotFoundError: module = None
    assert module is not None, 'independent episode qualifier/exporter is missing'
    return module


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


def ledger(n_frames=12, phase='HOLD'):
    sensor = HallSimulator(default_calibration())
    rows, frames = [], []
    for index in range(n_frames * 12):
        stamp = (index + 1) / 240.
        row = dict(time_s=stamp, q=np.zeros(7).tolist(), dq=np.zeros(7).tolist(),
                   q_command=[0., 0., .00001, 0., 0., 0., 0.],
                   compression_m=.001, phase=phase, target='R1', subphase='contact_cycle',
                   recovery=False, training_accepted=True)
        sensor.update(stamp, row['compression_m'])
        rows.append(row)
        if (index + 1) % 12 == 0:
            frames.append(dict(index=len(frames), time_s=stamp, physics_row=index,
                phase=phase, target='R1', subphase='contact_cycle', eligible=True,
                q=row['q'].copy(), dq=row['dq'].copy(), hall=asdict(sensor.read(stamp)),
                render_physics_delta_s=0., prompt=INSTRUCTIONS[phase]))
    return rows, frames


def test_sensor_replay_checks_every_physics_step_and_exact_packet():
    module = api(); rows, frames = ledger()
    result = module.replay_sensor_bindings(rows, frames)
    assert result['physics_rows_replayed'] == len(rows)
    assert result['hall_packets_matched'] == len(frames)
    # Row 10 is an ADC acquisition, between the 20 Hz captured frames.
    rows[10]['compression_m'] = .002
    with pytest.raises(ValueError, match='Hall'):
        module.replay_sensor_bindings(rows, frames)


@pytest.mark.parametrize('field', ['q', 'dq', 'time_s', 'hall', 'eligible'])
def test_sensor_ledger_tampering_is_rejected(field):
    module = api(); rows, frames = ledger()
    if field in ('q', 'dq'): frames[0][field][0] += .01
    elif field == 'time_s': frames[0][field] += .001
    elif field == 'hall': frames[0]['hall']['raw'] += 1
    else: frames[0]['eligible'] = False
    with pytest.raises(ValueError): module.replay_sensor_bindings(rows, frames)


def test_labels_use_executed_following_command_and_current_hall_not_future_feedback():
    module = api(); rows, frames = ledger(2, 'APPROACH')
    current, following = frames
    current['q'] = [0., -.2, -.3, .2, 0., 0., 0.]
    command = np.array(current['q']); command[3] += .002
    rows[following['physics_row']]['q_command'] = command.tolist()
    following['q'] = (command + .1).tolist()
    future_sensor = HallSimulator(default_calibration()); future_sensor.update(following['time_s'], .003)
    following['hall'] = asdict(future_sensor.read(following['time_s']))
    result = module.build_labels(rows, frames)
    action = result['actions'][0]
    kin = OpenArmKinematics()
    travel = default_calibration().decode(HallPacket(**current['hall']), now_s=current['time_s']).displacement_estimate_m
    origin, destination = kin.tcp(np.array(current['q']), travel), kin.tcp(command, travel)
    expected = np.r_[destination[:3, 3] - origin[:3, 3],
                      Rotation.from_matrix(destination[:3, :3] @ origin[:3, :3].T).as_rotvec(), 0.]
    np.testing.assert_allclose(action, expected, atol=1e-12)
    future_travel = default_calibration().decode(HallPacket(**following['hall']), now_s=following['time_s']).displacement_estimate_m
    incorrect = kin.tcp(command, future_travel)[:3, 3] - kin.tcp(np.array(current['q']), future_travel)[:3, 3]
    assert np.linalg.norm(action[:3] - incorrect) > 1e-7


@pytest.mark.parametrize('boundary', ['phase', 'target', 'recovery', 'ineligible'])
def test_labels_never_bridge_hidden_interval_boundaries(boundary):
    module = api(); rows, frames = ledger(3)
    if boundary == 'phase': rows[18]['phase'] = 'CONTACT'
    elif boundary == 'target': rows[18]['target'] = 'L1'
    elif boundary == 'recovery': rows[18]['recovery'] = True
    else: frames[1]['eligible'] = False
    result = module.build_labels(rows, frames)
    assert 0 not in result['frame_indices']
    assert result['diagnostics']['excluded_pairs'] > 0


def test_bounds_reject_whole_episode_without_clipping_and_report_phase_maxima():
    module = api(); rows, frames = ledger(3, 'RETRACT')
    rows[frames[1]['physics_row']]['q_command'][1] = -.01
    result = module.build_labels(rows, frames)
    assert result['diagnostics']['actions_bounded'] is False
    assert result['diagnostics']['phase_maxima']['RETRACT']['translation_norm_m'] > .001
    assert result['diagnostics']['violations'][0]['frame_index'] == 0
    assert np.linalg.norm(result['actions'][0, :3]) > .001


def episode_fixture(tmp_path, monkeypatch, *, audit_accepted=True):
    module = api()
    from sim.auscultation_vla_v2.tests.test_body_tracking import frame, K, POLY, SEEDS
    from sim.auscultation_vla_v2.body_tracking import BodyTracker
    from sim.auscultation_vla_v2.observations import Observation, RobotState, encode
    from sim.auscultation_skill_v1 import audits
    from sim.auscultation_skill_v1.media_teardown import POLICY
    episode = tmp_path / 'source'; episode.mkdir()
    sensors = episode / 'sensors'; sensors.mkdir()
    snapshot = episode / 'input_snapshot'; (snapshot / 'objects').mkdir(parents=True)
    files = {}
    def pin(key, path):
        data = Path(path).read_bytes(); digest = hashlib.sha256(data).hexdigest()
        (snapshot / 'objects' / digest).write_bytes(data)
        files[key] = dict(source_path=str(path), sha256=digest, bytes=len(data))
    for key, path in module.replay_source_paths().items(): pin(key, path)
    annotation = tmp_path / 'annotation.json'
    write_json(annotation, dict(target_pixels=SEEDS, support_polygon_px=POLY.tolist()))
    pin('source/vla/annotation', annotation)
    write_json(snapshot / 'manifest.json', dict(schema=1, files=files, points=['R1']))
    camera_source = tmp_path / 'approved_camera.json'
    write_json(camera_source, dict(intrinsics=dict(width=240, height=200, fx=220., fy=220., cx=120., cy=100.),
                                   world_from_optical=np.eye(4).tolist()))
    camera = dict(schema='wam_live_rgbd_capture_v1', width=240, height=200,
                  intrinsics=K.tolist(), camera_to_base=np.eye(4).tolist(),
                  camera_sha256=sha(camera_source), annotation_sha256=sha(annotation),
                  calibration_sha256=default_calibration().sha256, audio='absent',
                  depth_semantics='metric_optical_z', target_source='rgbd_surface_tracking')
    write_json(sensors / 'camera.json', camera)
    rows, frames = ledger()
    tracker, kin = BodyTracker(K, np.eye(4)), OpenArmKinematics()
    rgb, depth, valid = frame()
    for index, row in enumerate(frames):
        stamp = row['time_s']; times = dict(rgb_time_s=stamp, depth_time_s=stamp, now_s=stamp)
        estimate = (tracker.initialize(rgb, depth, valid, SEEDS, POLY, **times) if index == 0
                    else tracker.update(rgb, depth, valid, **times))['R1']
        raw = sensors / ('%05d.npz' % index)
        np.savez_compressed(raw, rgb=rgb, depth_m=depth, depth_valid=valid)
        wrist = kin.fk(np.array(row['q']))
        robot = RobotState(np.array(row['q']), np.array(row['dq']), wrist[:3, 3], wrist[:3, :3], HallPacket(**row['hall']), stamp)
        obs = Observation(rgb, depth, valid, K, np.eye(4), stamp, stamp, robot, estimate, INSTRUCTIONS[row['phase']])
        row.update(raw_file=raw.name, raw_sha256=sha(raw), state=encode(obs, now_s=stamp)['state'].tolist(),
                   target_estimate={key: value.tolist() if isinstance(value, np.ndarray) else value
                                    for key, value in asdict(estimate).items()})
    write_json(episode / 'control_commands.json', [])
    np.savez(episode / 'expert_replay.npz', test_only=np.zeros(1))
    write_json(episode / 'world_from_left_base.json', np.eye(4).tolist())
    outcome = dict(success=True, training_accepted=True, sensor_frames=len(frames), requested_points=['R1'],
                   points=[dict(target='R1', success=True, training_accepted=True)],
                   input_snapshot='input_snapshot/manifest.json', input_snapshot_manifest_sha256=sha(snapshot / 'manifest.json'))
    write_json(episode / 'outcome.json', outcome)
    teardown = dict(policy=POLICY, replicator_wait_completed=True, annotators_detached=2, render_products_destroyed=1,
                    capture_scope_ended=True, gc_before_stage_close=True, pre_stage_close_updates=5,
                    explicit_stage_close_checked=True, stage_absent_after_drain=True, post_stage_close_updates=5,
                    gc_before_app_close=True, close_returned=True, errors=[])
    write_json(episode / 'collection_result.json', dict(passed=True, sensor_frames=len(frames), teardown=teardown, errors=[]))
    def save_ledgers():
        for name, values in [('expert_trace.jsonl', rows), ('sensors/frames.jsonl', frames)]:
            (episode / name).write_text(''.join(json.dumps(value, allow_nan=False) + '\n' for value in values))
    save_ledgers()
    calls = []
    def independent_audit(path):
        calls.append(Path(path))
        return dict(accepted=audit_accepted, audits={
            'mechanics': dict(accepted=audit_accepted, artifact_sha256={name: sha(episode / name)
                for name in ('expert_trace.jsonl', 'control_commands.json', 'outcome.json')},
                control_trace_binding=dict(physics_rows_covered=len(rows))),
            'posture': dict(accepted=audit_accepted, physics_steps=len(rows), artifact_sha256={name: sha(episode / name)
                for name in ('expert_trace.jsonl', 'expert_replay.npz', 'outcome.json')},
                source_sha256=dict(kinematics_urdf=dict(sha256=sha(DEFAULT_URDF_PATH)),
                    world_from_left_base=dict(sha256=sha(episode / 'world_from_left_base.json'))))})
    monkeypatch.setattr(audits, 'audit_saved', independent_audit)
    return episode, camera_source, rows, frames, save_ledgers, calls


def test_verified_replay_exports_exact_trainer_arrays_and_hash_evidence(tmp_path, monkeypatch):
    module = api(); episode, camera, rows, frames, _, calls = episode_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'qualified'
    report = module.qualify_episode(episode, output, episode_id='cpu_fixture', camera_source_path=camera)
    assert calls == [episode]
    assert report['qualified'] is True and all(report['checks'].values())
    assert report['evidence']['raw_frames_verified'] == len(frames)
    assert report['evidence']['source_sha256'] and report['evidence']['camera_sha256'] == sha(camera)
    assert report['arrays_sha256'] == sha(output / 'cpu_fixture.npz')
    with np.load(output / 'cpu_fixture.npz', allow_pickle=False) as arrays:
        assert set(arrays.files) == {'state', 'actions', 'rgb', 'times', 'phases', 'targets', 'prompts', 'target_sources'}
        assert arrays['state'].shape == (11, 32) and arrays['actions'].shape == (11, 7)
        assert arrays['rgb'].shape == (11, 187, 224, 3)
        np.testing.assert_array_equal(arrays['state'][0], np.asarray(frames[0]['state'], np.float32))
        assert set(arrays['target_sources']) == {'rgbd_surface_tracking'}


@pytest.mark.parametrize('failure', ['v1_audit', 'raw_hash', 'state', 'oracle', 'camera', 'bounds'])
def test_failed_episode_gets_diagnostics_but_no_training_npz(tmp_path, monkeypatch, failure):
    module = api(); episode, camera, rows, frames, save, _ = episode_fixture(tmp_path, monkeypatch, audit_accepted=failure != 'v1_audit')
    if failure == 'raw_hash': frames[0]['raw_sha256'] = '0' * 64
    elif failure == 'state': frames[3]['state'][16] += .01
    elif failure == 'oracle': frames[0]['target_estimate']['source'] = 'fakeoracle_target'
    elif failure == 'camera':
        value = json.loads((episode / 'sensors/camera.json').read_text()); value['intrinsics'][0][0] += 1
        write_json(episode / 'sensors/camera.json', value)
    elif failure == 'bounds': rows[frames[1]['physics_row']]['q_command'][1] = -.01
    save()
    output = tmp_path / 'rejected'
    report = module.qualify_episode(episode, output, episode_id='cpu_fixture', camera_source_path=camera)
    assert report['qualified'] is False and report['reasons']
    assert not list(output.glob('*.npz'))
    assert (output / 'cpu_fixture.qualification.json').is_file()
    if failure == 'bounds':
        assert report['label_diagnostics']['phase_maxima']['HOLD']['translation_norm_m'] > .001


def test_saved_pass_flag_never_replaces_real_v1_audit(tmp_path):
    module = api(); episode = tmp_path / 'incomplete'; episode.mkdir()
    write_json(episode / 'collection_result.json', {'passed': True})
    # Actual V1 auditor executes on the absent evidence and rejects it.
    report = module.qualify_episode(episode, tmp_path / 'rejected', episode_id='cpu_fixture')
    assert report['qualified'] is False
    assert (episode / 'verification_summary.json').is_file()
