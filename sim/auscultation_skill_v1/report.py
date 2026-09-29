"""Build a portable, fail-closed Chinese review from actual saved evidence.

Run on the evidence's original host: existing quality/dataset validators require
their bound paths. The resulting review bundle is portable and serves no source
code, checkpoints, or arbitrary project files.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess

from .config import ROOT, PROJECT, TARGETS, ENGINEERING_GATES
from .contract import read, require, sha


def child(directory, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and all(part not in ('.', '..') and not part.startswith('.')
                    for part in relative.split('/')), 'unsafe evidence child path')
    directory = Path(directory).resolve()
    result = (directory / relative).resolve()
    require(directory in result.parents, 'evidence path escapes its directory')
    return result


def summarize_heldout(results, expected_conditions):
    require(bool(expected_conditions) and len(set(expected_conditions)) == len(expected_conditions),
            'expected heldout conditions must be unique and nonempty')
    by_condition = {}
    for result in results:
        name = result.get('condition_id')
        require(name in expected_conditions and name not in by_condition, 'unexpected or duplicate heldout condition')
        by_condition[name] = result
    trials = []
    for name in expected_conditions:
        result = by_condition.get(name, {})
        verified = result.get('evidence_verified') is True
        complete = (verified and result.get('requested_points') == list(TARGETS)
                    and result.get('complete_requested_sequence') is True)
        trials.append(dict(condition_id=name, evidence_verified=verified,
            learned_task_success=bool(complete and result.get('mode') == 'policy'
                                      and result.get('learned_task_success') is True),
            assisted_system_success=bool(complete and result.get('assisted_system_success') is True),
            state='missing' if not result else 'verified' if verified else 'invalid',
            reason=result.get('evidence_error'), metrics=result.get('metrics')))
    successes = sum(row['learned_task_success'] for row in trials)
    return dict(requested_trials=len(expected_conditions), learned_successes=successes,
                learned_success_rate=successes / len(expected_conditions),
                assisted_successes=sum(row['assisted_system_success'] for row in trials),
                missing_conditions=[name for name in expected_conditions if name not in by_condition],
                required_points=list(TARGETS), trials=trials)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'evidence hash mismatch: ' + path.name)
        require(path not in self.hashes or self.hashes[path] == digest, 'evidence changed during report')
        self.hashes[path] = digest
        return path

    def read(self, path, expected=None):
        return read(self.bind(path, expected))

    def quality(self, path):
        from .quality import _read_bound_quality
        self.bind(path)
        quality = _read_bound_quality(path)
        for name, digest in quality['artifact_sha256'].items():
            self.bind(child(quality['episode'], name), digest)
        for filename, digest in quality['audit_sources'].items():
            self.bind(filename, digest)
        legacy_binding = quality['inherited_reports']['legacy']
        legacy = self.read(legacy_binding['path'], legacy_binding['sha256'])
        free = quality['inherited_reports']['free_motion']
        self.bind(free['path'], free['sha256'])
        mechanical = legacy['contact']['legacy']['points']
        drag = legacy['contact']['loaded_drag']
        metrics = dict(points=mechanical,
            max_physical_force_N=max((p['max_physical_force_N'] for p in mechanical), default=None),
            minimum_continuous_hold_s=min((p['continuous_front_contact_hold_s'] for p in mechanical), default=None),
            command_peaks={name: entry['peak']['value'] for name, entry in quality['command_forward'].get('metrics', {}).items()},
            loaded_max_event_m=drag.get('max_event_distance_m'), loaded_max_speed_m_s=drag.get('max_speed_m_s'),
            loaded_events=drag.get('events', []), measured_raw_target_status=quality.get('measured_raw_target_status'),
            measured_raw_acceleration_peak_rad_s2=max(legacy['motion']['global']['measured']['derivatives']['raw']['acceleration']['max_abs_per_joint']),
            measured_raw_jerk_peak_rad_s3=max(legacy['motion']['global']['measured']['derivatives']['raw']['jerk']['max_abs_per_joint']),
            hard_simulation_gates_passed=quality.get('hard_simulation_gates_passed') is True,
            command_quality_passed=quality.get('command_quality_passed') is True,
            loaded_contact_quality_passed=quality.get('loaded_contact_quality_passed') is True,
            worker_process_gate_passed=quality.get('worker_process_gate_passed') is True)
        return quality, metrics

    def evaluation(self, path, model=None):
        result = self.read(path)
        require(result.get('schema') == 'auscultation_scalar_evaluation_v1', 'wrong evaluation schema')
        binding = result['quality_report']
        self.bind(binding['path'], binding['sha256'])
        quality, metrics = self.quality(binding['path'])
        require(Path(result['episode']).resolve() == Path(quality['episode']).resolve(), 'evaluation episode mismatch')
        require(result.get('condition_id') == quality.get('condition_id')
                and result.get('requested_points') == quality.get('requested_points'), 'evaluation condition/point request mismatch')
        require(set(result.get('artifact_sha256', {})) == {'expert_trace.jsonl', 'control_commands.json', 'outcome.json',
                                                          'runner_summary.json', 'worker_completion.json'},
                'evaluation trace/control/outcome/lifecycle binding incomplete')
        for name, digest in result['artifact_sha256'].items():
            require(quality['artifact_sha256'].get(name) == digest, 'evaluation artifact is not quality-bound')
            self.bind(child(result['episode'], name), digest)
        require(result.get('mode') in ('policy', 'zero', 'untrained'), 'unknown evaluation mode')
        if result['mode'] == 'policy':
            require(model is not None and result.get('checkpoint_sha256') == model['checkpoint_sha256']
                    and result.get('model_manifest_sha256') == model['manifest_sha256'], 'evaluation model mismatch')
        attribution = result.get('action_attribution', {})
        require(type(attribution.get('action_attribution_passed')) is bool
                and type(attribution.get('decisions_replayed')) is int
                and attribution['decisions_replayed'] >= 0, 'action attribution report is incomplete')
        if result.get('learned_task_success') is True:
            require(result['mode'] == 'policy' and result.get('student_identity_valid') is True
                    and quality.get('episode_simulation_gates_passed') is True
                    and attribution['action_attribution_passed'] is True
                    and attribution['decisions_replayed'] > 0
                    and attribution.get('hard_intervention_records') == 0, 'unsupported learned success claim')
        if result.get('assisted_system_success') is True:
            require(quality.get('hard_simulation_gates_passed') is True and result.get('student_identity_valid') is True,
                    'unsupported assisted completion claim')
        fresh = self.recheck_evaluation(result, binding['path'])
        require(fresh == result, 'saved evaluation differs from fresh model/source/lifecycle replay')
        return dict(result, evidence_verified=True, metrics=metrics)

    def recheck_evaluation(self, result, quality_path):
        """CPU-only frozen-model replay, including current source and lifecycle checks."""
        from .evaluate import audit_episode
        episode = Path(result['episode'])
        outcome = self.read(episode/'outcome.json')
        snapshot = self.read(child(episode, outcome['input_snapshot']), outcome['input_snapshot_manifest_sha256'])
        entry = snapshot['files']['source/contact_skill/learning/manifest']
        manifest_path = Path(entry['source_path']).resolve()
        require(PROJECT in manifest_path.parents, 'captured model is outside the authorized project')
        self.bind(manifest_path, result['model_manifest_sha256'])
        self.bind(manifest_path.parent/'policy.pt', result['checkpoint_sha256'])
        self.bind(ROOT/'evaluate.py')
        return audit_episode(episode, quality_path, manifest_path.parent, mode=result['mode'])

    def media_receipt(self, path):
        record = self.read(path)
        require(record.get('schema') == 'auscultation_skill_media_verification_v1'
                and record.get('status') == 'passed', 'media verification receipt did not pass')
        self.bind(ROOT/'verify_media.py', record['verifier_sha256'])
        self.bind(PROJECT/'sim/upper_pair_posture_20260924/verify_video.py', record['decoder_sha256'])
        from .verify_media import verify_render, verify_rgbd
        reports = record.get('reports')
        require(isinstance(reports, dict) and reports and set(reports) <= {'render', 'rgbd'},
                'media verification reports missing or unknown')
        for kind, result in reports.items():
            manifest = self.bind(result['manifest'], result['manifest_sha256'])
            process = result['process_completion']
            guard = self.bind(process['guard_path'], process['guard_sha256'])
            self.bind(process['launcher_path'], process['launcher_sha256'])
            verify = verify_render if kind == 'render' else verify_rgbd
            fresh = verify(manifest.parent, guard_path=guard)
            require(fresh == result, 'media verification differs from fresh source/process/content audit')
        return reports

    def stable_inventory(self):
        result = []
        for path, digest in self.hashes.items():
            require(sha(path) == digest, 'evidence changed before publishing: ' + path.name)
            label = str(path.relative_to(PROJECT)) if PROJECT in path.parents else path.name
            result.append(dict(source=label, sha256=digest))
        return result


def video_metadata(data, outcome):
    require(data.get('playback_speed') == 1., 'video must preserve original time at 1x')
    require(outcome.get('requested_points') == list(TARGETS), 'video must contain the complete ten-point request')
    intervals = data.get('point_intervals', [])
    require([p.get('target') for p in intervals] == list(TARGETS)
            and len(outcome.get('points', [])) == len(TARGETS), 'video point coverage is incomplete')
    count = data.get('source_physics_rows')
    require(type(count) is int and count > 0 and count == outcome.get('physics_steps'), 'video physics count mismatch')
    require(data.get('fps') == 15 and data.get('resolution') == [1920, 720], 'video format changed')
    dt = data.get('source_physics_dt_s')
    require(type(dt) in (float, int) and abs(dt - 1/240) < 1e-12, 'video physics clock changed')
    end = 0; points = []
    for record, executed in zip(intervals, outcome.get('points', [])):
        require(record == {k: executed[k] for k in ('target', 'start_physics_row', 'end_physics_row')},
                'video point intervals differ from outcome')
        require(record['start_physics_row'] == end and type(record['end_physics_row']) is int
                and end < record['end_physics_row'] <= count, 'video point intervals have gaps')
        end = record['end_physics_row']
        points.append(dict(target=record['target'], time_s=record['start_physics_row'] * dt))
    require(end == count and len(points) == 10, 'video omitted a requested point or final interval')
    frames = data.get('frames', [])
    expected_count = math.ceil(count / 16)
    require(len(frames) == expected_count and data.get('frame_count') == expected_count, 'video frame inventory incomplete')
    for index, frame in enumerate(frames):
        require(frame.get('frame_index') == index and abs(frame['video_time_s'] - index/15) < 1e-7
                and frame.get('measurement_row') == min((index+1)*16, count)-1,
                'video frame times were changed or omitted')
    require(frames[-1].get('subphase') == 'home_settle', 'video final home is missing')
    require(abs(data['source_duration_s'] - count*dt) < 1e-7
            and abs(data['duration_s'] - expected_count/15) < 1e-7, 'video duration mismatch')
    return dict(points=points, source_duration_s=data['source_duration_s'], duration_s=data['duration_s'],
                playback_speed=1., frame_count=expected_count, fps=15)


def bind_renderer_inventory(evidence, record):
    hashes = record.get('renderer_implementation_sha256', {})
    required = {'sim/auscultation_skill_v1/' + name for name in
                ('render.py', 'quality.py', 'provenance.py', 'contract.py')}
    require(isinstance(hashes, dict) and required.issubset(hashes), 'renderer implementation binding incomplete')
    for relative, digest in hashes.items():
        evidence.bind(child(PROJECT, relative), digest)


def build_report(output, *, qualify=None, collect=None, model=None, evaluate=None,
                 rgbd_manifest=None, expert_video_manifest=None, learned_video_manifest=None,
                 negative_evaluation=(), video_verification=(), rgbd_verification=(), conditions=ROOT/'conditions.json'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    evidence = Evidence(); components = {}; videos = {}; media_sources = {}; expert_episodes = set()
    condition_document = evidence.read(conditions)
    require(condition_document.get('schema') == 'auscultation_skill_conditions_v1', 'wrong condition manifest')
    expected = [row['condition_id'] for row in condition_document['conditions'] if row['split'] == 'test']
    heldout = summarize_heldout([], expected)

    def stage(name, operation):
        try:
            value = operation()
            components[name] = dict(status='passed', **value)
            return value
        except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.TimeoutExpired) as error:
            components[name] = dict(status='incomplete', reason=str(error))
            return None

    def qualification():
        require(qualify is not None, 'qualification campaign not supplied')
        from .quality import verify_repetitions
        path = Path(qualify)/'repetitions.json'; evidence.bind(path)
        record = verify_repetitions(path)
        pairs = [evidence.quality(e['path']) for e in record['episodes']]
        qualities = [pair[1] for pair in pairs]
        expert_episodes.update(Path(pair[0]['episode']).resolve() for pair in pairs)
        return dict(distinct_reset_count=record['distinct_reset_count'], episodes=qualities)
    stage('qualification', qualification)

    def collection():
        require(collect is not None, 'collection campaign not supplied')
        from .dataset import load_dataset
        directory = Path(collect)/'dataset'
        _, record = load_dataset(directory)
        evidence.bind(directory/'manifest.json'); evidence.bind(directory/'dataset.npz', record['arrays_sha256'])
        evidence.bind(directory/'conditions.json', record['conditions_sha256'])
        require(record['conditions_sha256'] == sha(conditions), 'dataset uses different reserved conditions')
        needed = {r['condition_id'] for r in condition_document['conditions'] if r['split'] in ('train', 'validation')}
        require({r['condition_id'] for r in record['episodes']} == needed, 'collection condition inventory incomplete')
        pairs = [evidence.quality(r['quality_report']) for r in record['episodes']]
        qualities = [pair[1] for pair in pairs]
        expert_episodes.update(Path(pair[0]['episode']).resolve() for pair in pairs)
        return dict(sample_count=record['sample_count'], episode_count=len(record['episodes']),
                    manifest_sha256=sha(directory/'manifest.json'), episodes=qualities)
    collected = stage('collection', collection)

    def training():
        require(model is not None and collected is not None, 'model or verified dataset unavailable')
        directory = Path(model); record = evidence.read(directory/'run_manifest.json')
        require(record.get('schema') == 'auscultation_scalar_training_v1', 'wrong model manifest')
        evidence.bind(directory/'policy.pt', record['checkpoint_sha256'])
        evidence.bind(directory/'dataset_manifest.json', record['dataset_manifest_sha256'])
        require(record['dataset_manifest_sha256'] == collected['manifest_sha256'], 'training dataset mismatch')
        require(record.get('selection') == 'validation MSE only' and record.get('test_evaluated') is False
                and record.get('split_episode_ids', {}).get('test') == [], 'training heldout isolation is not established')
        require(str(record.get('device', '')).startswith('cuda') and bool(record.get('history')), 'CUDA optimization evidence missing')
        require(record.get('conditions_sha256') == sha(conditions), 'training conditions changed')
        for name, digest in record['code_sha256'].items(): evidence.bind(child(ROOT, name), digest)
        return dict(checkpoint_sha256=record['checkpoint_sha256'], manifest_sha256=sha(directory/'run_manifest.json'),
                    device=record['device'], cuda_device_name=record.get('cuda_device_name'),
                    best_epoch=record['best_epoch'], architecture=record['architecture'], selection=record['selection'])
    trained = stage('training', training)

    results = []
    if evaluate is not None:
        directory = Path(evaluate)
        for name in expected:
            path = directory/'evaluations'/(name+'.json')
            if not path.exists(): continue
            try:
                result = evidence.evaluation(path, trained)
                require(result['condition_id'] == name, 'evaluation filename/condition mismatch')
                receipt = evidence.read(directory/(name+'.trial.json'))
                require(receipt.get('evaluation') == {k:v for k,v in result.items() if k not in ('evidence_verified', 'metrics')},
                        'campaign receipt differs from evaluation')
                results.append(result)
            except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, subprocess.TimeoutExpired) as error:
                results.append(dict(condition_id=name, evidence_verified=False, evidence_error=str(error)))
    heldout = summarize_heldout(results, expected)
    components['evaluation'] = dict(status='passed' if all(r['evidence_verified'] for r in heldout['trials']) else 'incomplete',
                                    verified_trials=sum(r['evidence_verified'] for r in heldout['trials']))

    def negatives():
        records = [evidence.evaluation(path) for path in negative_evaluation]
        require(len(records) == 2 and {r['mode'] for r in records} == {'zero', 'untrained'}, 'both zero and untrained controls are required')
        require(all(r.get('negative_control_task_completed') is False and r.get('learned_task_success') is False
                    and r.get('negative_control_diagnostic_passed') is True
                    and r['action_attribution']['action_attribution_passed'] is True
                    and r['action_attribution']['decisions_replayed'] > 0
                    and r['action_attribution'].get('hard_intervention_records') == 0
                    for r in records), 'a negative control completed the task')
        return dict(modes=[r['mode'] for r in records], both_failed_to_complete=True)
    stage('negative_controls', negatives)

    def rgbd():
        require(rgbd_manifest is not None, 'RGB-D sensor manifest not supplied')
        path = Path(rgbd_manifest); record = evidence.read(path)
        require(record.get('schema') == 'auscultation_saved_pose_rgbd_v1'
                and record.get('source_files_unchanged') is True and record.get('physics_advanced') is False
                and record.get('raw_rgb_has_overlays') is False, 'RGB-D acquisition evidence incomplete')
        for relative, digest in record['source_sha256'].items(): evidence.bind(child(PROJECT, relative), digest)
        from .rgbd import _calibration
        _calibration(record['intrinsics'], record['world_from_optical'])
        samples = record['samples']; require(len(samples) == record['sample_count'] and bool(samples), 'RGB-D samples missing')
        expected_samples = {(target, kind) for target in TARGETS for kind in ('prepose', 'hold', 'release')}
        require(expected_samples.issubset({(s['target'], s['kind']) for s in samples}), 'RGB-D point samples incomplete')
        projections = []
        for sample in samples:
            require(sample['rgb_source_time_s'] == sample['depth_source_time_s'] == sample['simulation_time_s'], 'RGB-D timestamps differ')
            require(set(sample['files']) == {'rgb', 'depth_m', 'valid_mask'}, 'RGB-D modalities missing')
            for name, relative in sample['files'].items(): evidence.bind(child(path.parent, relative), sample['file_sha256'][name])
            require({p['name'] for p in sample['target_projection']} == set(TARGETS), 'RGB-D target projection inventory incomplete')
            projections.extend(sample['target_projection'])
        receipts = [evidence.media_receipt(item).get('rgbd') for item in rgbd_verification]
        matches = [item for item in receipts if item and item.get('manifest_sha256') == sha(path)]
        require(len(matches) == 1, 'exactly one verification of the exact RGB-D manifest is required')
        # media_receipt already repeated the complete source/process/array/projection audit.
        return dict(sample_count=len(samples), depth_matches=sum(p['depth_matches_within_5mm'] for p in projections),
                    target_projection_count=len(projections), minimum_valid_depth_fraction=min(s['valid_depth_fraction'] for s in samples),
                    live_policy_input=False, image_encoder_trained=False)
    stage('rgbd_interface', rgbd)

    def video(role, path):
        require(path is not None, role+' video manifest not supplied')
        path = Path(path); record = evidence.read(path)
        require(record.get('schema') == 'auscultation_skill_recorded_full10_review_v1'
                and record.get('source_files_unchanged') is True, 'video source evidence incomplete')
        expected_class = ('sim.auscultation_skill_v1.controller.ExpertController' if role == 'expert'
                          else 'sim.auscultation_skill_v1.student.StudentController')
        require(record.get('executed_controller_class') == expected_class, 'video controller role mismatch')
        for relative, digest in record['source_sha256'].items(): evidence.bind(child(PROJECT, relative), digest)
        bind_renderer_inventory(evidence, record)
        episode = child(PROJECT, record['episode']); outcome = evidence.read(episode/'outcome.json')
        if role == 'expert':
            require(episode in expert_episodes, 'expert video is not a qualified campaign episode')
        else:
            require(trained is not None and outcome.get('policy_mode') == 'policy'
                    and outcome.get('policy_checkpoint_sha256') == trained['checkpoint_sha256'],
                    'learned video did not execute the reviewed checkpoint')
            require(any(r.get('evidence_verified') is True and Path(r['episode']).resolve() == episode
                        for r in results), 'learned video is not an audited reserved trial')
        data = video_metadata(record, outcome)
        media = evidence.bind(child(path.parent, record['video']), record['video_sha256'])
        require(media.suffix == '.mp4', 'review video must be MP4')
        data.update(src='media/'+role+'.mp4', manifest_sha256=sha(path), video_sha256=record['video_sha256'],
                    fidelity_verified=False)
        videos[role] = data; media_sources[data['src']] = media
        return dict(manifest_sha256=sha(path), video_sha256=record['video_sha256'])
    stage('expert_video', lambda: video('expert', expert_video_manifest))
    stage('learned_video', lambda: video('learned', learned_video_manifest))

    def fidelity():
        records = []
        for path in video_verification:
            receipt = evidence.media_receipt(path)
            if 'render' in receipt: records.append(receipt['render'])
        require(len(videos) == 2, 'both review videos required')
        for data in videos.values():
            matches = [r for r in records if r.get('manifest_sha256') == data['manifest_sha256']
                       and r.get('video_sha256') == data['video_sha256']]
            require(len(matches) == 1, 'missing or duplicate verification for exact video and manifest')
            row = matches[0]
            require(row.get('status') == 'passed' and row.get('decode', {}).get('decoded_through_eof') is True
                    and row['decode'].get('pts_verified') is True
                    and row.get('schedule', {}).get('all_frame_source_bindings_verified') is True,
                    'media decoder/frame attribution evidence incomplete')
            data['fidelity_verified'] = True
        return dict(original_time_videos_verified=2)
    stage('media_verification', fidelity)

    complete = all(value['status'] == 'passed' for value in components.values())
    all_metrics = [m for key in ('qualification', 'collection') for m in components.get(key, {}).get('episodes', [])]
    all_metrics += [row['metrics'] for row in heldout['trials'] if row.get('metrics')]
    measured_peaks = [m['measured_raw_jerk_peak_rad_s3'] for m in all_metrics if m.get('measured_raw_jerk_peak_rad_s3') is not None]
    acceleration_peaks = [m['measured_raw_acceleration_peak_rad_s2'] for m in all_metrics
                          if m.get('measured_raw_acceleration_peak_rad_s2') is not None]
    measured = dict(target_rad_s3=ENGINEERING_GATES['measured_raw_jerk_target_rad_s3'],
                    observed_peak_rad_s3=max(measured_peaks, default=None),
                    acceleration_target_rad_s2=ENGINEERING_GATES['joint_acceleration_rad_s2'],
                    observed_acceleration_peak_rad_s2=max(acceleration_peaks, default=None),
                    status='exceeded' if any(m.get('measured_raw_target_status') == 'exceeded' for m in all_metrics)
                           else 'within_recorded_bounds' if all_metrics else 'unavailable')
    result = dict(schema='auscultation_skill_review_v1', generated_utc=datetime.now(timezone.utc).isoformat(),
        evidence_complete=complete, simulation_v1_success=bool(complete and heldout['learned_successes'] == heldout['requested_trials']),
        external_acceptance='pending_not_evaluated', components=components, heldout=heldout, videos=videos,
        measured_raw_derivative_limitation=measured,
        scope=dict(policy='state BC; learned desired normal velocity on the checked contact curve',
                   routing='explicit free-space planner and ten-point scheduler', rgbd='calibrated saved-pose sensor interface',
                   image_encoder_trained=False, rgbd_live_policy_input=False, acoustic_reward=False,
                   real_hardware_tested=False, clinical_acceptance=False),
        limitations=['固定场景中的状态行为克隆；未证明跨患者、几何或相机泛化。',
                     'RGB-D 为保存姿态的传感器接口验证，未训练图像编码器，也未用于策略闭环输入。',
                     '已完成 V4 专家试验的日志包含 PhysX 材质面索引查询警告；材质、摩擦及 solver 是否受影响尚未验证，不能将警告视为无害。',
                     '噪声示范覆盖的是 actor Hall 观察下的双向 HOLD 修正；CONTACT 阶段尚无高于目标力的卸压示范，不能据此声称任意接触扰动均已覆盖。',
                     '实测原始关节导数目标单独披露；尚无声学、临床或真实硬件验收。'],
        evidence=evidence.stable_inventory())
    output.mkdir(parents=True, exist_ok=False)
    for relative, source in media_sources.items():
        destination = output/relative; destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha(destination) == evidence.hashes[source], 'copied media hash mismatch')
    shutil.copyfile(ROOT/'review.html', output/'review.html')
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    names = ['review.html', 'summary.json', *media_sources]
    manifest = dict(schema='auscultation_skill_review_bundle_v1',
                    files={name: {'sha256': sha(output/name), 'bytes': (output/name).stat().st_size} for name in names})
    (output/'serve_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qualify', 'collect', 'model', 'evaluate', 'rgbd-manifest', 'expert-video-manifest', 'learned-video-manifest'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--negative-evaluation', type=Path, action='append', default=[])
    parser.add_argument('--video-verification', type=Path, action='append', default=[])
    parser.add_argument('--rgbd-verification', type=Path, action='append', default=[])
    parser.add_argument('--conditions', type=Path, default=ROOT/'conditions.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if ROOT not in args.output.resolve().parents: parser.error('review output must stay inside the new skill package')
    result = build_report(**vars(args))
    print(json.dumps({key:result[key] for key in ('evidence_complete', 'simulation_v1_success')}))
    return 0 if result['simulation_v1_success'] else 1


if __name__ == '__main__': raise SystemExit(main())
