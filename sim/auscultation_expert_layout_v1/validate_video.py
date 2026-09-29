"""Verify both expert video views against accepted recordings and decoded frames.

Run as a module with --episode and --video-dir. This verifies recording integrity;
the bound independent mechanical and posture audits establish simulation acceptance.
It imports no Isaac Sim code and never renders or advances the physics timeline.
"""
import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from .render_expert import (
    accepted_frame, check_camera_reference, front_force, playback_frames,
    point_dwell_manifest, point_records, read_camera_reference,
    recorded_pose_digest, recorded_schedule_digest, validate_low_elbow_export,
    validate_replay_timing, validate_scene_binding, verify_reference_inputs,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_frame_schedule(rows, indices, frames, fps, motion_speed, view_mode):
    """Bind every display interval to its source measurement and nearest raw pose."""
    require(fps == 15 and motion_speed in (1, 2), 'unsupported playback timing')
    require(view_mode in ('robotcam', 'posture', 'report'), 'unsupported view mode')
    times, dt, pose_period = validate_replay_timing(rows, indices)
    first_pose_row = int(indices[0])
    plan = ([{'measurement_row': first_pose_row, 'speed': 0, 'shot': 'overview'}] * (2 * fps)
            + playback_frames(rows, first_pose_row, fps, max_motion_speed=motion_speed)
            + [{'measurement_row': len(rows) - 1, 'speed': 0, 'shot': 'overview'}] * (2 * fps))
    expected = [(spec['measurement_row'], spec['speed'],
                 view_mode if view_mode != 'report' else spec['shot']) for spec in plan]
    actual = [(frame['measurement_row'], frame['playback_speed'], frame['shot']) for frame in frames]
    require(actual == expected, 'recorded playback schedule changed')
    require([frame['video_frame'] for frame in frames] == list(range(len(frames))),
            'video frame indices changed')
    for frame in frames:
        row = frame['measurement_row']
        sample = rows[row]
        pose_index = int(np.searchsorted(indices, row, side='right') - 1)
        require(pose_index >= 0 and frame['pose_index'] == pose_index,
                'pose is not the nearest preceding recorded sample')
        pose_row = int(indices[pose_index])
        require(frame['pose_row'] == pose_row, 'recorded pose row changed')
        age = float(times[row] - times[pose_row])
        require(frame['pose_age_s'] == age, 'recorded pose age changed')
        require(-1e-8 <= age <= pose_period + dt + 1e-7,
                'pose and measurement timestamps are too far apart')
        for key, expected_value in (
            ('target', sample['target']), ('phase', sample['phase']),
            ('simulation_time_s', sample['time_s']), ('hold_elapsed_s', sample['hold_elapsed_s']),
            ('physical_front_N', front_force(sample)), ('membrane_N', sample['diaphragm_N']),
        ):
            require(frame[key] == expected_value, 'frame measurement changed: ' + key)
        if frame['phase'] == 'HOLD':
            require(frame['playback_speed'] == 1, 'HOLD must play at 1x')


def validate_manifest(episode, manifest):
    """Validate captured bytes, acceptance, exact playback and camera metadata."""
    require(manifest['preview'] is False, 'preview is not a complete recording')
    names = manifest['inputs_sha256']
    require(all(Path(name).name == name for name in names), 'invalid input filename')
    inputs = {name: (episode / name).read_bytes() for name in names}
    hashes = {name: sha(data) for name, data in inputs.items()}
    require(hashes == names, 'source recording changed')
    rows = [json.loads(line) for line in inputs['expert_trace.jsonl'].splitlines() if line.strip()]
    outcome = json.loads(inputs['outcome.json'])
    evidence = {name: json.loads(inputs[name]) for name in (
        'independent_audit.json', 'posture_audit.json', 'verification_summary.json', 'runner_summary.json')}
    require(validate_low_elbow_export(outcome, rows, False, evidence, hashes), 'episode rejected')
    validate_scene_binding(outcome, hashes)
    expected_targets = [side + str(level) for level in range(1, 6) for side in ('R', 'L')]
    require(outcome['requested_points'] == expected_targets, 'complete ordered ten-point recording required')
    camera, reference_hashes = read_camera_reference(episode, outcome)
    require(reference_hashes == manifest['camera_reference_inputs_sha256'], 'camera reference hashes changed')
    check_camera_reference(manifest['camera_reference'], camera)
    view_mode = manifest['view_mode']
    require(manifest['robotcam_view'] is (view_mode == 'robotcam'), 'RobotCam view label mismatch')
    require(manifest['posture_view'] is (view_mode == 'posture'), 'posture view label mismatch')
    require(manifest['render_camera_path'] == (
        '/World/RobotCam' if view_mode == 'robotcam' else '/World/RecordedReplayReportCamera'),
        'render camera does not match view label')
    require(manifest['resolution'] == ([camera['width'], camera['height']] if view_mode == 'robotcam'
                                        else [1280, 720]), 'video resolution changes camera aspect ratio')
    for flag in ('training_accepted', 'source_layers_unchanged', 'physics_timeline_stopped',
                 'original_rgbd_camera_unchanged', 'camera_pose_optics_verified_every_frame'):
        require(manifest[flag] is True, 'missing required evidence: ' + flag)
    require(manifest['physics_timeline_mode'] == 'stopped_static_pose_replay'
            and manifest['pose_application'] == 'recorded_raw_rigid_body_transforms_no_interpolation',
            'recorded pose/static timeline disclosure changed')
    with np.load(io.BytesIO(inputs['expert_replay.npz']), allow_pickle=False) as replay:
        poses, indices, paths = replay['poses'], replay['row_indices'], replay['paths']
    require(poses.shape == (len(indices), len(paths), 7) and np.isfinite(poses).all(),
            'invalid pose recording')
    frames = manifest['frames']
    require(len(frames) == manifest['frame_count'], 'manifest frame count changed')
    validate_frame_schedule(rows, indices, frames, manifest['fps'],
                            manifest['motion_playback_speed_cap'], view_mode)
    require(manifest['playback_schedule_sha256'] == recorded_schedule_digest(frames),
            'playback schedule hash changed')
    require(manifest['recorded_poses'] == recorded_pose_digest(poses, [f['pose_index'] for f in frames]),
            'recorded pose hash changed')
    records = point_records(rows, outcome)
    for frame in frames:
        require(frame['training_accepted'] is accepted_frame(rows[frame['measurement_row']],
                                                             records[frame['target']], outcome),
                'frame acceptance differs from audited episode')
    points = point_dwell_manifest(rows, frames, records, manifest['fps'])
    require(set(points) == set(manifest['points']), 'video target set changed')
    for target, point in points.items():
        point.update(recorded_pose_digest(poses, [f['pose_index'] for f in frames if f['target'] == target]))
        point['source_trace_sha256'] = hashes['expert_trace.jsonl']
        point['source_replay_sha256'] = hashes['expert_replay.npz']
        require(point == manifest['points'][target], 'dwell or pose evidence changed: ' + target)
        require(point['hold_frames_at_1x'] == 45 and point['hold_video_duration_s'] == 3.,
                'each point must retain the actual three-second HOLD')
    require(manifest['hold_frames_at_1x'] == 450, 'ten-point HOLD frame count changed')
    return points, hashes, reference_hashes


def probe_and_decode(video, manifest, ffprobe='ffprobe', ffmpeg='ffmpeg'):
    """Use independent media tools to count and decode every actual video frame."""
    probe = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)['streams']
    require(len(streams) == 1, 'expected exactly one selected video stream')
    stream = streams[0]
    require(int(stream['nb_read_frames']) == manifest['frame_count'], 'decoded frame count mismatch')
    require([stream['width'], stream['height']] == manifest['resolution'], 'decoded resolution mismatch')
    require(Fraction(stream['avg_frame_rate']) == manifest['fps'], 'decoded frame rate mismatch')
    subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    episode, directory = args.episode.resolve(), args.video_dir.resolve()
    manifest_path = directory / 'video_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    points, hashes, references = validate_manifest(episode, manifest)
    videos = list(directory.glob('*_recorded.mp4'))
    require(len(videos) == 1, 'expected exactly one recorded video in the view directory')
    video = videos[0]
    video_before = sha(video.read_bytes())
    require(manifest['video_sha256'] == video_before, 'video differs from renderer output')
    stream = probe_and_decode(video, manifest, args.ffprobe, args.ffmpeg)
    require(video_before == sha(video.read_bytes()), 'video changed during validation')
    require(manifest_bytes == manifest_path.read_bytes(), 'manifest changed during validation')
    require(all(sha((episode / name).read_bytes()) == value for name, value in hashes.items()),
            'recording changed during validation')
    verify_reference_inputs(episode, references)
    report = {'decoded_all_frames': True, 'source_hashes_rechecked': True, 'pose_hashes_rechecked': True,
        'schedule_hash_rechecked': True, 'camera_reference_hashes_rechecked': True,
        'simulation_accepted': True, 'frame_count': int(stream['nb_read_frames']), 'fps': manifest['fps'],
        'resolution': manifest['resolution'], 'duration_s': manifest['frame_count'] / manifest['fps'],
        'total_hold_frames_1x': sum(point['hold_frames_at_1x'] for point in points.values()),
        'total_hold_video_s': sum(point['hold_video_duration_s'] for point in points.values()),
        'view_mode': manifest['view_mode'], 'render_camera_path': manifest['render_camera_path'],
        'motion_playback_speed_cap': manifest['motion_playback_speed_cap'], 'video_sha256': video_before,
        'video_manifest_sha256': sha(manifest_bytes), 'video_bytes': video.stat().st_size,
        'scope': 'Decoded recording integrity and bound simulation acceptance; no acoustic or clinical validation.',
        'points': {target: {'hold_frames_1x': point['hold_frames_at_1x'], 'hold_video_s': point['hold_video_duration_s']}
                   for target, point in points.items()}}
    with (directory / 'video_validation.json').open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'points'}))


if __name__ == '__main__':
    main()
