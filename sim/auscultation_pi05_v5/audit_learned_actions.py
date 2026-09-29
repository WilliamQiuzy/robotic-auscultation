"""CPU-only audit of saved joint proposals, sensor bindings and executed ticks.

Passing establishes consistency of the recorded command evidence. It never
accepts a policy, contact mechanics, collision safety, or task completion.
Preparation, recovery and final home are identified but not scored as learned.
"""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import PIL
from PIL import Image

from .execution import JointTrajectory


DT = 1 / 240
COMMAND_ATOL = 1e-10
ORIGIN_ATOL = 1e-8
PHASES = {'FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT'}
BINDING_FIELDS = ('checkpoint_metadata_sha256', 'data_hash', 'inference_source_sha256',
                  'sampling_steps', 'sampling_seed', 'sampling_policy', 'action_semantics',
                  'training_recipe', 'modalities', 'audio')
FINGERPRINTS = ('request_payload_sha256', 'rgb_pixel_sha256', 'rgb_shape',
                'state_f32_sha256', 'raw_action_f32_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), 'Invalid ' + name)
    return result


def close(actual, expected, name, atol=COMMAND_ATOL):
    actual = vector(actual, np.shape(expected), name)
    require(np.allclose(actual, expected, atol=atol, rtol=0), name + ' mismatch')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def f32_digest(value):
    return digest(np.asarray(value, dtype='<f4').tobytes())


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def request_digest(rgb, state, phase, target, request_id):
    """Reconstruct the versioned client encoding from independently saved inputs."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    body = dict(request_id=request_id, phase=phase, target=target,
                state=np.asarray(state).tolist(), rgb_png=base64.b64encode(buffer.getvalue()).decode())
    return digest(json.dumps(body).encode())


def verify_sensor(record, frame, receipt, raw_sensor_root, report):
    require(record['raw_sensor_sha256'] == frame['raw_sha256'], 'Raw sensor/frame hash mismatch')
    state = vector(frame['state'], (32,), 'frame state')
    require(receipt.get('state_f32_sha256') == f32_digest(state), 'state_f32_sha256 mismatch')
    for key in FINGERPRINTS:
        if key != 'rgb_shape':
            require(valid_hash(receipt.get(key)), 'Missing or invalid fingerprint: ' + key)
    if raw_sensor_root is None:
        report['limitations'].append('Raw sensor NPZ unavailable for frame ' + str(frame['index']))
        return False
    root = Path(raw_sensor_root).resolve()
    relative = frame['raw_file']
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Invalid raw sensor path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Raw sensor path escapes sensor root')
    if not path.is_file():
        report['limitations'].append('Raw sensor NPZ unavailable: ' + str(path))
        return False
    data = path.read_bytes()
    require(digest(data) == frame['raw_sha256'], 'Raw sensor file hash mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        rgb = archive['rgb']
    require(rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8, 'Invalid saved RGB pixels')
    require(receipt.get('rgb_shape') == list(rgb.shape), 'rgb_shape mismatch')
    require(receipt.get('rgb_pixel_sha256') == digest(rgb.tobytes()), 'rgb_pixel_sha256 mismatch')
    require(receipt['request_payload_sha256'] == request_digest(
        rgb, frame['state'], record['phase'], record['target'], receipt['request_id']),
        'request_payload_sha256 mismatch; exact PNG/JSON encoder must match the bound client')
    report['raw_sensor_files_sha256'][relative] = frame['raw_sha256']
    return True


def _audit(learned, frames, physics, *, binding, expected_checkpoint_metadata_sha256,
           raw_sensor_root, server_requests, velocity, task_scope, episode_failed, report):
    require(valid_hash(expected_checkpoint_metadata_sha256), 'Explicit expected checkpoint hash required')
    require(binding.get('checkpoint_metadata_sha256') == expected_checkpoint_metadata_sha256,
            'Expected checkpoint differs from saved policy binding')
    require(all(key in binding for key in BINDING_FIELDS), 'Incomplete policy binding')
    require(binding['action_semantics'] == 'measured_joint_command_offset_v1'
            and binding['training_recipe'] == 'phase_balanced_measured_joint_actions_v5'
            and binding['sampling_steps'] == 10 and binding['sampling_seed'] == 917
            and binding['audio'] == 'absent', 'Incompatible joint policy binding')
    require(np.isfinite(velocity) and 0 < velocity <= .25, 'Invalid explicit Ruckig velocity limit')
    require(task_scope in {'full', 'contact'}, 'Unknown task scope')
    require([row['index'] for row in learned] == list(range(len(learned))), 'Learned query indices are not contiguous')
    require([row['index'] for row in frames] == list(range(len(frames))), 'Sensor frame indices are not contiguous')
    times = vector([row['time_s'] for row in physics], (len(physics),), 'physics clock')
    require(len(times) > 0 and np.allclose(np.diff(times), DT, atol=1e-8, rtol=0), 'Physics clock is discontinuous')
    request_map = {}
    if server_requests is not None:
        for request in server_requests:
            identity = request['request_id']
            require(identity not in request_map, 'Duplicate server request ID')
            request_map[identity] = request
    seen_requests, seen_targets = set(), set()
    trajectory = None
    previous_end, previous_target, previous_phase = 0, None, None
    previous_failure = False
    release_stopped = False
    prediction_count = sensor_count = 0

    def replay(row, expected, kind):
        close(row['q_command'], expected, 'Ruckig ' + kind + ' q_command')

    def gap(start, end, *, initial=False, final=False):
        nonlocal release_stopped
        if start == end:
            return
        names = Counter()
        release_steps = 0
        stopping = False
        for index in range(start, end):
            row = physics[index]
            subphase = row.get('subphase')
            names[subphase] += 1
            if initial:
                allowed = {'initial_home', 'settle'}
                if task_scope == 'contact':
                    allowed |= {'transfer', 'align', 'lower_approach', 'inter_point_transfer'}
                require(subphase in allowed and row.get('recovery') is False,
                        'Unexplained preparation gap at physics row ' + str(index))
            elif subphase == 'learned_release_stop':
                require(trajectory is not None and previous_phase == 'RETRACT' and not previous_failure
                        and row['target'] == previous_target and row.get('recovery') is False,
                        'Unbound learned release-stop gap')
                require(not release_stopped, 'Repeated release-stop after stationary completion')
                if not stopping:
                    require(index > 0, 'Missing release-stop origin')
                    trajectory.set_target(vector(physics[index - 1]['q_command'], (7,), 'release origin'))
                    stopping = True
                release_steps += 1
                require(release_steps <= 120, 'Release-stop exceeds 120 physics ticks')
                replay(row, trajectory.step(), 'release-stop')
                report['verified_release_stop_steps'] += 1
                release_stopped = (np.max(abs(trajectory.velocity)) < 1e-10
                                   and np.max(abs(trajectory.acceleration)) < 1e-10)
            elif subphase == 'withdraw_settle':
                require(release_stopped and row['target'] == previous_target and row.get('recovery') is False,
                        'Withdrawal gap lacks a verified stationary release')
                close(row['q_command'], vector(physics[index - 1]['q_command'], (7,), 'previous command'),
                      'Stationary withdrawal command')
            elif final and subphase in {'return_bridge', 'return_home', 'home_settle'}:
                require(release_stopped and row.get('recovery') is False, 'Final home lacks verified learned release')
            elif final and subphase == 'fault_retract' and row.get('recovery') is True:
                require(previous_failure or episode_failed, 'Recovery gap without recorded episode failure')
            else:
                raise ValueError('Unexplained command gap at physics row ' + str(index) + ': ' + str(subphase))
        report['non_query_intervals'].append(dict(start=start, end_exclusive=end,
            role='initial_preparation' if initial else 'post_query_release_or_finalization', subphase_counts=dict(names)))

    for index, record in enumerate(learned):
        report['current_query'] = index
        require(not previous_failure, 'Query follows an exception; retained generator state cannot be verified')
        start, end = record['physics_row_start'], record['physics_row_end_exclusive']
        require(type(start) is int and type(end) is int and 0 < start <= end <= len(physics), 'Invalid query physics interval')
        require(start >= previous_end, 'Overlapping or reordered query physics intervals')
        require(end - start <= 12, 'Query executes more than 12 physics ticks')
        phase, target = record['phase'], record['target']
        require(phase in PHASES and isinstance(target, str) and target, 'Invalid query phase/target')
        require(record.get('expert_fallback') is False, 'Expert fallback is not learned execution')
        failure = bool(record.get('failure_reason'))
        if end - start < 12:
            require(failure or (end > start and (record.get('prefix_interrupted_for_phase_change') is True
                    or record.get('prefix_interrupted_for_target_revision') is True)), 'Unexplained partial query prefix')
        gap(previous_end, start, initial=index == 0)
        frame_index = record['observation_frame']
        require(type(frame_index) is int and 0 <= frame_index < len(frames), 'Invalid observation frame reference')
        frame = frames[frame_index]
        require(frame['physics_row'] == start - 1 and frame['target'] == target, 'Observation frame/time/target mismatch')
        close(record['time_s'], times[start - 1], 'Query time', atol=1e-8)
        close(frame['time_s'], times[start - 1], 'Observation frame time', atol=1e-8)
        measured = vector(frame['q'], (7,), 'frame measured q')
        close(measured, vector(physics[start - 1]['q'], (7,), 'physics measured q'), 'Frame measured origin', ORIGIN_ATOL)
        close(vector(frame['state'], (32,), 'frame state')[:7], measured, 'Frame state measured origin', ORIGIN_ATOL)
        require(record['raw_sensor_sha256'] == frame['raw_sha256'], 'Raw sensor frame binding mismatch')
        has_prediction = 'raw_actions' in record or 'inference' in record
        if not has_prediction:
            require(failure and end == start and record.get('accepted') is False,
                    'Missing prediction on executed or accepted query')
            report['unverified_requests'] += 1
            report['limitations'].append('Query ' + str(index) + ' failed before a saved prediction receipt')
        else:
            receipt = record['inference']
            for key in BINDING_FIELDS:
                require(receipt.get(key) == binding[key], 'Inference ' + key + ' binding mismatch')
            raw = vector(record['raw_actions'], (10, 7), 'raw action chunk')
            require(np.array_equal(raw, vector(receipt['raw_actions'], (10, 7), 'receipt raw actions')),
                    'Saved raw actions differ from inference receipt')
            require(receipt.get('raw_action_f32_sha256') == f32_digest(raw), 'raw_action_f32_sha256 mismatch')
            identity = receipt['request_id']
            require(isinstance(identity, str) and identity and identity not in seen_requests, 'Invalid or reused request ID')
            seen_requests.add(identity)
            if server_requests is not None:
                require(identity in request_map, 'Inference request absent from server log')
                request = request_map[identity]
                require(request['phase'] == phase and request['target'] == target
                        and request['checkpoint_metadata_sha256'] == expected_checkpoint_metadata_sha256,
                        'Server request phase/target/checkpoint mismatch')
                for key in FINGERPRINTS:
                    require(request.get(key) == receipt.get(key), 'Server request ' + key + ' mismatch')
            sensor_count += int(verify_sensor(record, frame, receipt, raw_sensor_root, report))
            prediction_count += 1
            require(record.get('executed_chunk_prefix') == 1, 'Expected first chunk action only')
            if 'measured_q' in record or 'goal_q' in record:
                close(record['measured_q'], measured, 'Logged measured origin', ORIGIN_ATOL)
                goal = measured + raw[0]
                close(record['goal_q'], goal, 'Raw measured-relative joint goal')
            else:
                require(failure and end == start and record.get('accepted') is False,
                        'Accepted/executed query lacks measured origin and goal')
                goal = None
            if record.get('accepted') is True:
                require(goal is not None, 'Accepted query lacks goal')
                limit = .007 if phase in {'CONTACT', 'HOLD', 'RETRACT'} else .035
                require(np.all(abs(raw[0]) <= limit + 1e-12), 'Executed raw joint proposal exceeds phase bound')
                if trajectory is None or target != previous_target:
                    require(target not in seen_targets, 'Repeated target violates unique sequence/reset contract')
                    if trajectory is not None:
                        require(release_stopped, 'New target reset lacks verified release continuity')
                    require(start >= 3, 'Insufficient stationary command history for zero-derivative initialization')
                    seed = vector(physics[start - 1]['q_command'], (7,), 'initial retained q_command')
                    for previous in physics[start - 3:start]:
                        close(previous['q_command'], seed, 'Stationary initial command history')
                    trajectory = JointTrajectory(seed, dt=DT, velocity=velocity, acceleration=.5, jerk=8.)
                    report['trajectory_initializations'] += 1
                    seen_targets.add(target)
                    release_stopped = False
                trajectory.set_target(goal)
                for row in physics[start:end]:
                    require(row['target'] == target and row.get('subphase') == 'learned_' + phase.lower()
                            and row.get('recovery') is False, 'Executed row is not the logged learned query')
                    replay(row, trajectory.step(), 'query')
                    report['verified_executed_steps'] += 1
            else:
                require(failure and end == start, 'Unaccepted query executed physics')
        report['queries_checked'] += 1
        report['failed_queries'] += int(failure)
        report['query_prefixes'].append(dict(index=index, start=start, end_exclusive=end,
                                             executed_steps=end-start, recorded_failure=failure))
        previous_end, previous_target, previous_phase = end, target, phase
        previous_failure = failure
    gap(previous_end, len(physics), initial=not learned, final=True)
    report.pop('current_query', None)
    report['command_binding_verified'] = report['verified_executed_steps'] > 0
    report['sensor_payload_binding_verified'] = prediction_count > 0 and sensor_count == prediction_count
    report['request_receipt_binding_verified'] = prediction_count > 0 and not report['unverified_requests']
    report['server_request_log_verified'] = server_requests is not None and prediction_count > 0
    report['evidence_consistent'] = True
    report['passed'] = (report['command_binding_verified'] and report['sensor_payload_binding_verified']
                        and report['request_receipt_binding_verified'])
    if server_requests is None:
        report['limitations'].append('Server request log unavailable; input hashes verified against saved inference receipts only')
    if not report['verified_executed_steps']:
        report['limitations'].append('No learned physics ticks were executed; no executed-command acceptance')


def audit_records(learned, frames, physics, *, binding, expected_checkpoint_metadata_sha256,
                  raw_sensor_root=None, server_requests=None, velocity=.25, task_scope='full', episode_failed=False):
    import ruckig
    report = dict(schema='wam_learned_joint_action_evidence_audit_v1', passed=False,
        evidence_consistent=False, command_binding_verified=False, sensor_payload_binding_verified=False,
        request_receipt_binding_verified=False, server_request_log_verified=False,
        policy_acceptance='not_evaluated', errors=[], limitations=[], queries_checked=0,
        failed_queries=0, unverified_requests=0, verified_executed_steps=0,
        verified_release_stop_steps=0, trajectory_initializations=0, query_prefixes=[],
        non_query_intervals=[], raw_sensor_files_sha256={},
        expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
        replay=dict(dt_s=DT, joint_velocity_rad_s=velocity, joint_acceleration_rad_s2=.5,
                    joint_jerk_rad_s3=8., command_atol_rad=COMMAND_ATOL, measured_origin_atol_rad=ORIGIN_ATOL,
                    ruckig_version=getattr(ruckig, '__version__', 'unavailable'), pillow_version=PIL.__version__),
        scope='Saved multimodal request/proposal/executed-command consistency only; independent physics audits decide policy acceptance')
    try:
        _audit(learned, frames, physics, binding=binding,
               expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
               raw_sensor_root=raw_sensor_root, server_requests=server_requests, velocity=velocity,
               task_scope=task_scope, episode_failed=episode_failed, report=report)
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as error:
        where = 'query ' + str(report['current_query']) + ': ' if 'current_query' in report else ''
        report['errors'].append(where + type(error).__name__ + ': ' + str(error))
    return report


def audit_episode(directory, *, expected_checkpoint_metadata_sha256, binding_path=None, server_requests_path=None,
                  raw_sensor_root=None):
    directory = Path(directory)
    sources = {}

    def read(path, rows=False):
        path = Path(path)
        data = path.read_bytes()
        sources[str(path.resolve())] = digest(data)
        return [json.loads(line) for line in data.splitlines()] if rows else json.loads(data)

    outcome = read(directory / 'outcome.json')
    snapshot_relative = outcome['input_snapshot']
    require(isinstance(snapshot_relative, str) and not Path(snapshot_relative).is_absolute(),
            'Invalid runtime snapshot path')
    snapshot_path = (directory / snapshot_relative).resolve()
    require(snapshot_path.is_relative_to(directory.resolve()), 'Runtime snapshot escapes episode')
    snapshot = read(snapshot_path)
    require(sources[str(snapshot_path)] == outcome['input_snapshot_manifest_sha256'],
            'Runtime snapshot manifest hash mismatch')
    runtime_sha = snapshot['files']['source/pi05_v5/execution.py']['sha256']
    require(valid_hash(runtime_sha), 'Invalid runtime execution snapshot hash')
    runtime_object = snapshot_path.parent / 'objects' / runtime_sha
    runtime_bytes = runtime_object.read_bytes()
    sources[str(runtime_object)] = digest(runtime_bytes)
    require(digest(runtime_bytes) == runtime_sha
            and digest(Path(__file__).with_name('execution.py').read_bytes()) == runtime_sha,
            'Replay execution differs from bound runtime snapshot')
    binding = read(binding_path) if binding_path is not None else outcome['learned_policy_binding']
    learned = read(directory / 'learned_actions.jsonl', rows=True)
    frames = read(directory / 'sensors/frames.jsonl', rows=True)
    physics = read(directory / 'expert_trace.jsonl', rows=True)
    requests = read(server_requests_path, rows=True) if server_requests_path is not None else None
    limits = outcome['learned_command_limits']
    require(limits['joint_acceleration_rad_s2'] == .5 and limits['joint_jerk_rad_s3'] == 8.,
            'Saved execution limits differ from the versioned auditor')
    report = audit_records(learned, frames, physics, binding=binding,
        expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
        raw_sensor_root=directory / 'sensors' if raw_sensor_root is None else raw_sensor_root,
        server_requests=requests, velocity=limits['joint_velocity_rad_s'],
        task_scope=outcome['task_scope'], episode_failed=outcome.get('success') is False)
    report['artifact_sha256'] = sources
    report['runtime_execution_snapshot_verified'] = True
    report['audit_source_sha256'] = {name: digest(Path(__file__).with_name(name).read_bytes())
                                    for name in ('audit_learned_actions.py', 'execution.py')}
    for path, expected in sources.items():
        if digest(Path(path).read_bytes()) != expected:
            report['passed'] = report['evidence_consistent'] = report['command_binding_verified'] = False
            report['errors'].append('Evidence changed during audit: ' + path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-checkpoint-metadata-sha256', required=True)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--server-requests', type=Path)
    parser.add_argument('--raw-sensor-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_episode(args.directory,
        expected_checkpoint_metadata_sha256=args.expected_checkpoint_metadata_sha256,
        binding_path=args.binding, server_requests_path=args.server_requests, raw_sensor_root=args.raw_sensor_root)
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
