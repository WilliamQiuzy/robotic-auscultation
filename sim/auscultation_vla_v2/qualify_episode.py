"""Independent CPU qualification of a completed collect_expert/live_capture run.

Re-runs the V1 mechanics/posture audits, then replays every 240 Hz Hall update,
every saved RGB-D frame through BodyTracker, and measured-joint FK/encode32.
Recorded target estimates are comparison evidence only: never tracker inputs.
Teacher q_command supplies labels only, using CURRENT measured q/Hall as origin.
No acoustic input, simulator target pose, force-truth feature, clipping, phase
relabeling, reseeding, interpolation, or failed-episode training fallback exists.

The new output directory receives a qualification JSON even on rejection, but
an exact eight-array trainer NPZ only after all checks pass. Raw RGB is tracked
at capture resolution; export defaults to an explicitly recorded 224-pixel-wide
aspect-preserving RGB resize. The trainer performs letterboxing. No train/val
manifest is generated. Simulation qualification is not physical/clinical proof.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from sim.auscultation_skill_v1 import audits
from sim.auscultation_skill_v1.media_teardown import require_complete_teardown
from sim.auscultation_expert_layout_v1.kinematics import DEFAULT_URDF_PATH, OpenArmKinematics
from .body_tracking import BodyTracker
from .chunks import PHASES
from .action_contract import limits as action_limits,bounded,VERSION as ACTION_CONTRACT_VERSION
from .demo_labels import label_pair
from .hall import HallPacket, HallSimulator, default_calibration
from .multimodal import INSTRUCTIONS
from .observations import Observation, RobotState, encode, finite, rotation
from .training_data import MODALITIES, full_chunks

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA = PROJECT / 'sim/camera_framing_20260923/recommended_camera.json'
EXCLUDED_SUBPHASES = {'settle', 'initial_home', 'return_bridge', 'return_home', 'home_settle'}
FREE_SUBPHASES = {'transfer', 'inter_point_transfer', 'align', 'settle', 'initial_home'}
ARTIFACTS = ('expert_trace.jsonl', 'control_commands.json', 'outcome.json',
             'expert_replay.npz', 'world_from_left_base.json', 'collection_result.json',
             'sensors/camera.json', 'sensors/frames.jsonl')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _read(path): return _json(Path(path).read_bytes())


def _rows(path):
    result = [_json(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    if not result or not all(isinstance(row, dict) for row in result):
        raise ValueError('empty or malformed JSONL ledger')
    return result


def _relative(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('invalid evidence path')
    path = root / value
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('evidence must be a regular file inside episode')
    return path


def replay_source_paths():
    """Only sources present in the producer snapshot are required to match it.

    This new auditor/demo-label exporter is separately hashed, never falsely
    claimed to have existed in an older collection snapshot.
    """
    source = Path(__file__).parent
    paths = {'source/vla/' + name: source / name for name in
             ('collect_expert.py', 'live_capture.py', 'body_tracking.py', 'rigid_registration.py',
              'tracking.py', 'observations.py', 'hall.py', 'multimodal.py')}
    runtime = DEFAULT_URDF_PATH.parent
    paths.update({'source/code/kinematics.py': runtime / 'kinematics.py',
                  'source/code/tool_frames.py': runtime / 'tool_frames.py', 'source/urdf': DEFAULT_URDF_PATH})
    return paths


def _frame_contract(physics):
    """The producer's phase mapping, without changing saved phase labels."""
    subphase = physics.get('subphase')
    if not isinstance(subphase, str) or not subphase or type(physics.get('recovery')) is not bool:
        raise ValueError('missing physics subphase or recovery identity')
    eligible = subphase not in EXCLUDED_SUBPHASES and not physics['recovery']
    if subphase in FREE_SUBPHASES: phase = 'FREE'
    elif subphase.startswith('normal_'): phase = 'RETRACT'
    else: phase = physics.get('phase')
    if phase not in PHASES: phase, eligible = 'FREE', False
    return phase, eligible


def replay_sensor_bindings(physics, frames):
    """Default ADC simulation over ALL physics rows; exact saved packet binding."""
    if not physics or not frames:
        raise ValueError('empty sensor or physics ledger')
    times = np.asarray([row['time_s'] for row in physics], float)
    if (not np.isfinite(times).all() or not np.isclose(times[0], 1/240., atol=1e-7, rtol=0)
            or not np.allclose(np.diff(times), 1/240., atol=1e-7, rtol=0)):
        raise ValueError('physics timestamp coverage is not 240 Hz from origin')
    expected_rows = list(range(11, len(physics), 12))
    if ([row.get('physics_row') for row in frames] != expected_rows
            or any(type(row.get('physics_row')) is not int for row in frames)
            or [row.get('index') for row in frames] != list(range(len(frames)))):
        raise ValueError('sensor capture has missing, repeated, or reordered physics rows')
    sensor = HallSimulator(default_calibration())
    by_physics = {row['physics_row']: row for row in frames}
    for index, physics_row in enumerate(physics):
        stamp = float(physics_row['time_s'])
        sensor.update(stamp, float(physics_row['compression_m']))
        if index not in by_physics: continue
        frame = by_physics[index]
        if frame.get('time_s') != physics_row['time_s']:
            raise ValueError('sensor time differs from same physics row')
        for key in ('q', 'dq'):
            actual = finite(frame.get(key), (7,), 'sensor ' + key)
            expected = finite(physics_row.get(key), (7,), 'physics ' + key)
            if not np.array_equal(actual, expected):
                raise ValueError('sensor ' + key + ' differs from same physics row')
        if frame.get('hall') != asdict(sensor.read(stamp)):
            raise ValueError('Hall ADC packet differs from full physics replay at frame ' + str(frame['index']))
        default_calibration().decode(HallPacket(**frame['hall']), now_s=stamp)
        if (frame.get('target') != physics_row.get('target')
                or frame.get('subphase') != physics_row.get('subphase')
                or type(frame.get('eligible')) is not bool):
            raise ValueError('sensor target/subphase/eligibility identity mismatch')
        candidates = {_frame_contract(physics_row)}
        # env.step records the old manager phase. LiveCapture records the phase
        # after manager.update. At a transition only the immediate successor's
        # manager phase can explain that difference; such pairs are not labels.
        if index + 1 < len(physics) and physics[index+1].get('target') == physics_row.get('target'):
            candidates.add(_frame_contract(dict(physics_row, phase=physics[index+1].get('phase'))))
        if (frame.get('phase'), frame['eligible']) not in candidates:
            raise ValueError('saved phase/eligibility differs from producer physics contract')
        if frame.get('prompt') != INSTRUCTIONS[frame['phase']]:
            raise ValueError('saved task instruction differs from phase')
        delta = float(finite(frame.get('render_physics_delta_s'), (), 'render time delta'))
        if abs(delta) > 1e-8:
            raise ValueError('render advanced unaccounted physics')
    return dict(physics_rows_replayed=len(physics), hall_packets_matched=len(frames),
                measured_state_rows_matched=len(frames), hall_model='HallSimulator default parameters',
                sample_period_s=.01, seed=0, noise_std_v=0., voltage_bias_v=0.,
                drift_v_s=0., latency_s=0., curve_warp=0.)


def build_labels(physics, frames):
    """Produce unmodified raw labels and diagnose bounds over every eligible pair.

    Call replay_sensor_bindings first for certification. This pure diagnostic
    helper never makes a qualification decision or clips a rejected action.
    """
    actions, frame_indices, pairs, excluded, maxima, violations = [], [], [], [], {}, []
    for current, following in zip(frames[:-1], frames[1:]):
        start, end = current['physics_row'], following['physics_row']
        reason = None
        if not current.get('eligible') or not following.get('eligible'):
            reason = 'ineligible_endpoint'
        elif (current.get('phase') != following.get('phase') or current.get('target') != following.get('target')
              or not np.isclose(following['time_s'] - current['time_s'], .05, atol=1e-7, rtol=0)
              or end - start != 12):
            reason = 'phase_target_or_time_boundary'
        else:
            interval = physics[start:end+1]
            if (len(interval) != 13 or len({row.get('phase') for row in interval}) != 1
                    or any(row.get('target') != current['target'] or row.get('recovery') is not False
                           or _frame_contract(row) != (current['phase'], True) for row in interval)):
                reason = 'physics_interval_crosses_phase_target_recovery_or_ineligible_region'
        if reason:
            excluded.append(dict(frame_index=current['index'], reason=reason))
            continue
        command = finite(physics[end].get('q_command'), (7,), 'executed following command')
        action = label_pair(current, following, command)
        phase = current['phase']
        translation, angle, force = float(np.linalg.norm(action[:3])), float(np.linalg.norm(action[3:6])), float(action[6])
        metric = maxima.setdefault(phase, dict(label_count=0, translation_norm_m=0., rotation_norm_rad=0.,
                                               force_reference_N=0., absolute_channel_max=[0.] * 7))
        metric['label_count'] += 1
        metric['translation_norm_m'] = max(metric['translation_norm_m'], translation)
        metric['rotation_norm_rad'] = max(metric['rotation_norm_rad'], angle)
        metric['force_reference_N'] = max(metric['force_reference_N'], abs(force))
        metric['absolute_channel_max'] = np.maximum(metric['absolute_channel_max'], np.abs(action)).tolist()
        limits = action_limits(phase)
        if not bounded(action[None],phase):
            violations.append(dict(frame_index=current['index'], phase=phase, action=action.tolist(), limits=limits))
        actions.append(action); frame_indices.append(current['index'])
        pairs.append(dict(frame_index=current['index'], following_frame_index=following['index'],
                          physics_interval_inclusive=[start, end], command_physics_row=end))
    matrix = np.asarray(actions, float).reshape(-1, 7)
    return dict(actions=matrix, frame_indices=frame_indices, pairs=pairs,
                diagnostics=dict(label_count=len(actions), excluded_pairs=len(excluded), exclusions=excluded,
                                 phase_maxima=maxima, actions_bounded=not violations, violations=violations,
                                 action_origin='current measured q and current replay-verified Hall packet',
                                 command_source='following.physics_row executed expert_trace.q_command'))


def _bind_audit(report, hashes, count):
    if report.get('accepted') is not True:
        raise ValueError('independent V1 mechanics/posture audit rejected the episode')
    for name, artifacts in [('mechanics', ('expert_trace.jsonl', 'control_commands.json', 'outcome.json')),
                            ('posture', ('expert_trace.jsonl', 'expert_replay.npz', 'outcome.json'))]:
        result = report.get('audits', {}).get(name, {})
        if result.get('accepted') is not True or any(result.get('artifact_sha256', {}).get(key) != hashes[key] for key in artifacts):
            raise ValueError('independent V1 ' + name + ' audit artifact binding mismatch')
    mechanics, posture = report['audits']['mechanics'], report['audits']['posture']
    if (mechanics.get('control_trace_binding', {}).get('physics_rows_covered') != count
            or posture.get('physics_steps') != count
            or posture.get('source_sha256', {}).get('kinematics_urdf', {}).get('sha256') != _sha(DEFAULT_URDF_PATH)
            or posture.get('source_sha256', {}).get('world_from_left_base', {}).get('sha256') != hashes['world_from_left_base.json']):
        raise ValueError('independent V1 control/pose/source audit binding mismatch')


def _source_and_camera(episode, outcome, camera_source_path, annotation_path):
    snapshot_manifest = _relative(episode, outcome['input_snapshot'])
    if _sha(snapshot_manifest) != outcome['input_snapshot_manifest_sha256']:
        raise ValueError('input snapshot manifest binding mismatch')
    snapshot = _read(snapshot_manifest)
    for key, item in snapshot['files'].items():
        digest = item['sha256']
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid snapshot object identity')
        object_path = _relative(snapshot_manifest.parent, 'objects/' + digest)
        if _sha(object_path) != digest or object_path.stat().st_size != item['bytes']:
            raise ValueError('snapshot object hash/size mismatch: ' + key)
    sources = {}
    for key, source in replay_source_paths().items():
        digest = _sha(source)
        if snapshot['files'].get(key, {}).get('sha256') != digest:
            raise ValueError('replay source differs from collection snapshot: ' + key)
        sources[str(source)] = digest
    sources[str(Path(__file__))] = _sha(__file__)
    for source in (Path(__file__).with_name('demo_labels.py'), Path(audits.__file__),
                   Path(__file__).with_name('contract.json'), Path(__file__).with_name('training_data.py'),
                   Path(__file__).with_name('action_contract.py')):
        sources[str(source)] = _sha(source)
    camera_metadata = _read(episode / 'sensors/camera.json')
    camera_source_path = Path(camera_source_path).resolve()
    if _sha(camera_source_path) != camera_metadata.get('camera_sha256'):
        raise ValueError('approved camera hash differs from capture')
    approved = _read(camera_source_path)
    annotation_record = snapshot['files']['source/vla/annotation']
    annotation_object = snapshot_manifest.parent / 'objects' / annotation_record['sha256']
    annotation_path = annotation_object if annotation_path is None else Path(annotation_path).resolve()
    if _sha(annotation_path) != annotation_record['sha256'] or _sha(annotation_path) != camera_metadata.get('annotation_sha256'):
        raise ValueError('initial annotation hash differs from capture snapshot')
    annotation = _read(annotation_path)
    if (camera_metadata.get('schema') != 'wam_live_rgbd_capture_v1'
            or camera_metadata.get('calibration_sha256') != default_calibration().sha256
            or camera_metadata.get('audio') != 'absent' or camera_metadata.get('depth_semantics') != 'metric_optical_z'
            or camera_metadata.get('target_source') != 'rgbd_surface_tracking'):
        raise ValueError('invalid captured camera/Hall/target modality contract')
    width, height = camera_metadata['width'], camera_metadata['height']
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('invalid capture dimensions')
    original = approved['intrinsics']; scale = width / original['width']
    if height != round(original['height'] * scale):
        raise ValueError('capture dimensions are not approved aspect-ratio scale')
    intrinsic = np.array([[original['fx'] * scale, 0., original['cx'] * scale],
                          [0., original['fy'] * scale, original['cy'] * scale], [0., 0., 1.]])
    base = finite(_read(episode / 'world_from_left_base.json'), (4, 4), 'world_from_base')
    rotation(base[:3, :3], 'world_from_base')
    if not np.allclose(base[3], [0., 0., 0., 1.], atol=1e-8, rtol=0): raise ValueError('nonrigid base calibration')
    transform = np.linalg.inv(base) @ finite(approved['world_from_optical'], (4, 4), 'approved optical transform')
    if (not np.allclose(finite(camera_metadata['intrinsics'], (3, 3), 'captured intrinsics'), intrinsic, atol=1e-10, rtol=0)
            or not np.allclose(finite(camera_metadata['camera_to_base'], (4, 4), 'captured calibration'), transform, atol=1e-10, rtol=0)):
        raise ValueError('captured calibration differs from approved camera and audited base')
    seeds = {name: np.asarray(pixel, float) * scale for name, pixel in annotation['target_pixels'].items()}
    polygon = np.asarray(annotation['support_polygon_px'], float) * scale
    evidence = dict(source_sha256=sources, snapshot_manifest_sha256=_sha(snapshot_manifest),
                    camera_sha256=_sha(camera_source_path), capture_camera_sha256=_sha(episode / 'sensors/camera.json'),
                    annotation_sha256=_sha(annotation_path), calibration_sha256=default_calibration().sha256)
    bound_files = {snapshot_manifest: _sha(snapshot_manifest), camera_source_path: _sha(camera_source_path),
                   annotation_path: _sha(annotation_path), **{Path(path): digest for path, digest in sources.items()}}
    return camera_metadata, intrinsic, transform, seeds, polygon, evidence, bound_files


def _compare_estimate(recorded, replayed):
    if (not isinstance(recorded, dict) or recorded.get('source') != 'rgbd_surface_tracking'
            or type(recorded.get('revision')) is not int or recorded['revision'] != replayed.revision
            or recorded.get('time_s') != replayed.time_s
            or not np.isclose(float(recorded.get('confidence', np.nan)), replayed.confidence, atol=1e-8, rtol=0)):
        raise ValueError('recorded target provenance/time/revision/confidence differs from RGB-D replay')
    for key, shape in [('position_base_m', (3,)), ('rotation_base', (3, 3)), ('velocity_base_m_s', (3,))]:
        if not np.allclose(finite(recorded.get(key), shape, key), getattr(replayed, key), atol=1e-5, rtol=0):
            raise ValueError('recorded target ' + key + ' differs from RGB-D replay')


def qualify_episode(episode_dir, output_dir, *, episode_id, camera_source_path=DEFAULT_CAMERA,
                    annotation_path=None, rgb_width=224):
    """Write a diagnostic report; export NPZ only after independent all-pass replay."""
    if not isinstance(episode_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,80}', episode_id):
        raise ValueError('invalid episode identifier')
    if rgb_width is not None and (type(rgb_width) is not int or rgb_width < 2):
        raise ValueError('RGB export width must be an integer >=2 or None for raw')
    episode, output = Path(episode_dir).resolve(), Path(output_dir).resolve()
    if output == episode or output.is_relative_to(episode):
        raise ValueError('qualification output must be separate from source episode')
    output.mkdir(parents=True, exist_ok=False)
    checks = {key: False for key in ('rgbd_tracking_valid', 'measured_state_valid', 'hall_valid', 'synchronized',
              'actions_bounded', 'episode_complete', 'no_oracle_target', 'v1_audits_valid', 'source_binding_valid', 'camera_valid')}
    report = dict(schema='wam_vla_episode_qualification_v1', episode_id=episode_id,
                  data_kind='closed_loop_rollout', qualified=False, arrays_sha256=None,
                  modalities=dict(MODALITIES), checks=checks, reasons=[], evidence={}, label_diagnostics={},
                  action_contract_version=ACTION_CONTRACT_VERSION,
                  action_contract_sha256=_sha(Path(__file__).with_name('action_contract.py')),
                  scope='independently replayed simulation evidence; not physical, acoustic, or clinical validation')
    stage = 'independent_v1_audit'
    try:
        before = {name: _sha(episode / name) for name in ARTIFACTS if (episode / name).is_file()}
        # Never consume collection_result.passed or a previously saved audit as
        # acceptance. The actual V1 implementation reruns and writes its audits.
        independent = audits.audit_saved(episode)
        report['evidence']['independent_v1_audit'] = independent
        if independent.get('accepted') is not True:
            raise ValueError('independent V1 mechanics/posture audit rejected the episode')
        physics, frames = _rows(episode / 'expert_trace.jsonl'), _rows(episode / 'sensors/frames.jsonl')
        hashes = {name: _sha(_relative(episode, name)) for name in ARTIFACTS}
        if before != hashes: raise ValueError('source artifacts changed during independent audit')
        _bind_audit(independent, hashes, len(physics)); checks['v1_audits_valid'] = True
        report['evidence']['artifact_sha256'] = hashes
        outcome, collection = _read(episode / 'outcome.json'), _read(episode / 'collection_result.json')
        stage = 'episode_completion'
        require_complete_teardown(collection.get('teardown'), annotators=2, products=1)
        requested = outcome.get('requested_points')
        if (outcome.get('success') is not True or outcome.get('training_accepted') is not True
                or outcome.get('sensor_frames') != len(frames) or collection.get('sensor_frames') != len(frames)
                or collection.get('errors') != [] or not requested
                or [point.get('target') for point in outcome.get('points', [])] != requested
                or any(point.get('success') is not True or point.get('training_accepted') is not True for point in outcome['points'])):
            raise ValueError('episode did not complete accepted points and all sensor captures')
        checks['episode_complete'] = True
        stage = 'sensor_physics_replay'
        report['evidence']['sensor_replay'] = replay_sensor_bindings(physics, frames)
        checks['hall_valid'] = checks['synchronized'] = True
        stage = 'action_labels'
        labels = build_labels(physics, frames)
        report['label_diagnostics'] = labels['diagnostics']
        checks['actions_bounded'] = labels['diagnostics']['actions_bounded']
        if not checks['actions_bounded']:
            report['reasons'].append('action_labels: raw expert labels exceed existing action bounds; no clipping applied')
        if not labels['frame_indices']: raise ValueError('no eligible synchronized action pairs')
        report['evidence']['label_pairs'] = labels['pairs']
        stage = 'source_camera_binding'
        metadata, intrinsic, transform, seeds, polygon, evidence, bound_files = _source_and_camera(
            episode, outcome, camera_source_path, annotation_path)
        report['evidence'].update(evidence)
        checks['source_binding_valid'] = checks['camera_valid'] = True
        raw_names = [row.get('raw_file') for row in frames]
        if raw_names != ['%05d.npz' % index for index in range(len(frames))]:
            raise ValueError('raw frame identity is not exact capture sequence')
        if {path.name for path in (episode / 'sensors').glob('*.npz')} != set(raw_names):
            raise ValueError('unlogged or missing raw RGB-D captures')
        tracker, kin = BodyTracker(intrinsic, transform), OpenArmKinematics()
        selected = set(labels['frame_indices']); selected_states, selected_images = [], []
        verified_raw, maximum_state_error = [], 0.
        raw_shape = (metadata['height'], metadata['width'])
        width = raw_shape[1] if rgb_width is None else min(rgb_width, raw_shape[1])
        export_shape = (max(1, round(raw_shape[0] * width / raw_shape[1])), width)
        stage = 'rgbd_target_and_state_replay'
        for row in frames:
            raw_path = _relative(episode / 'sensors', row['raw_file'])
            raw = raw_path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            if digest != row.get('raw_sha256'):
                raise ValueError('raw RGB-D hash mismatch at frame ' + str(row['index']))
            bound_files[raw_path] = digest
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != {'rgb', 'depth_m', 'depth_valid'}:
                    raise ValueError('raw frame contains unsupported/oracle modalities')
                rgb, depth, valid = archive['rgb'], archive['depth_m'], archive['depth_valid']
            if rgb.shape != (*raw_shape, 3) or depth.shape != raw_shape or valid.shape != raw_shape:
                raise ValueError('raw RGB-D dimensions differ from bound calibration')
            stamp = row['time_s']; times = dict(rgb_time_s=stamp, depth_time_s=stamp, now_s=stamp)
            estimates = (tracker.initialize(rgb, depth, valid, seeds, polygon, **times) if row['index'] == 0
                         else tracker.update(rgb, depth, valid, **times))
            estimate = estimates[row['target']]
            _compare_estimate(row.get('target_estimate'), estimate)
            q, dq = np.asarray(row['q']), np.asarray(row['dq']); wrist = kin.fk(q)
            robot = RobotState(q, dq, wrist[:3, 3], wrist[:3, :3], HallPacket(**row['hall']), stamp)
            observation = Observation(rgb, depth, valid, intrinsic, transform, stamp, stamp, robot, estimate, INSTRUCTIONS[row['phase']])
            state = encode(observation, now_s=stamp, hall_calibration=default_calibration())['state']
            recorded = finite(row.get('state'), (32,), 'recorded 32-dimensional state')
            error = float(np.max(np.abs(recorded - state))); maximum_state_error = max(maximum_state_error, error)
            if not np.allclose(recorded, state, atol=1e-5, rtol=0):
                raise ValueError('recorded state differs from measured RGB-D/Hall/FK replay at frame ' + str(row['index']))
            verified_raw.append(dict(index=row['index'], raw_file=row['raw_file'], sha256=digest))
            if row['index'] in selected:
                image = rgb.copy() if export_shape == raw_shape else np.asarray(Image.fromarray(rgb).resize(
                    (export_shape[1], export_shape[0]), Image.Resampling.BILINEAR))
                selected_states.append(state); selected_images.append(image)
        report['evidence'].update(raw_frames_verified=len(verified_raw), raw_frames=verified_raw,
            maximum_state_replay_abs_error=maximum_state_error, state_comparison_atol=1e-5,
            rgb_export=dict(source_height_width=list(raw_shape), export_height_width=list(export_shape),
                            resize='aspect-preserving BILINEAR' if export_shape != raw_shape else 'none',
                            letterbox='deferred to trainer'))
        checks['rgbd_tracking_valid'] = checks['measured_state_valid'] = checks['no_oracle_target'] = True
        chosen = [frames[index] for index in labels['frame_indices']]
        arrays = dict(state=np.asarray(selected_states, np.float32), actions=labels['actions'].astype(np.float32),
                      rgb=np.stack(selected_images).astype(np.uint8), times=np.array([row['time_s'] for row in chosen], float),
                      phases=np.array([row['phase'] for row in chosen]), targets=np.array([row['target'] for row in chosen]),
                      prompts=np.array([INSTRUCTIONS[row['phase']] for row in chosen]),
                      target_sources=np.array(['rgbd_surface_tracking'] * len(chosen)))
        chunks = full_chunks(arrays['times'], arrays['phases'].tolist(), arrays['targets'].tolist(), arrays['actions'], horizon=10, period_s=.05)
        report['evidence']['full_ten_step_chunks'] = len(chunks)
        if not chunks: raise ValueError('no complete ten-step same-phase/target supervision chunks')
        stage = 'final_content_binding'
        for name, digest in hashes.items():
            if _sha(episode / name) != digest: raise ValueError('source artifact changed during replay: ' + name)
        for path, digest in bound_files.items():
            if _sha(path) != digest: raise ValueError('bound raw/source/camera evidence changed during replay: ' + str(path))
        if all(checks.values()) and not report['reasons']:
            destination = output / (episode_id + '.npz'); temporary = output / (episode_id + '.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, destination)
            report.update(qualified=True, arrays_sha256=_sha(destination))
    except Exception as error:
        report['reasons'].append(stage + ': ' + type(error).__name__ + ': ' + str(error))
    report_path = output / (episode_id + '.qualification.json')
    temporary_report = output / (episode_id + '.qualification.partial')
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary_report, report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--camera-source', type=Path, default=DEFAULT_CAMERA)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--rgb-width', type=int, default=224, help='aspect-preserving RGB export width')
    args = parser.parse_args(argv)
    report = qualify_episode(args.episode, args.output, episode_id=args.episode_id,
                             camera_source_path=args.camera_source, annotation_path=args.annotation, rgb_width=args.rgb_width)
    print(json.dumps(dict(qualified=report['qualified'], reasons=report['reasons'],
                         label_diagnostics=report['label_diagnostics']), indent=2, allow_nan=False))
    return 0 if report['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
