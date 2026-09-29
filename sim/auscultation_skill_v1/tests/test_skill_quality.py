"""Whole-trace continuity and research-only eligibility tests."""
import copy
import importlib
import json

import numpy as np
import pytest


def quality():
    try:
        return importlib.import_module('sim.auscultation_skill_v1.quality')
    except ModuleNotFoundError as error:
        pytest.fail('V1 implementation missing: ' + str(error))


def rows(commands):
    return [{'q_command': q.tolist(), 'time_s': (i+1)/240., 'target': 'R1',
             'phase': 'APPROACH' if i < 5 else 'CONTACT',
             'subphase': 'lower_approach' if i < 5 else 'contact_cycle'}
            for i, q in enumerate(commands)]


def test_stationary_complete_trace_passes_forward_gate():
    report = quality().analyze_command_differences(rows(np.zeros((12, 7))), np.zeros(7))
    assert report['status'] == 'passed'
    assert report['sample_count'] == 12
    assert report['metrics']['jerk_rad_s3']['evaluated_windows'] == 10
    assert report['metrics']['jerk_rad_s3']['mixed_boundary_windows'] > 0
    assert report['excluded_window_count'] == 0


def test_mixed_contact_boundary_jerk_is_not_excluded():
    commands = np.zeros((12, 7))
    commands[5:, 0] = 1e-4
    report = quality().analyze_command_differences(rows(commands), np.zeros(7))
    assert report['status'] == 'failed'
    assert report['metrics']['jerk_rad_s3']['mixed_boundary_violation_windows'] > 0
    assert report['metrics']['jerk_rad_s3']['peak']['value'] > 20.


def test_initial_retained_command_and_timestamps_cannot_be_assumed():
    audit = quality()
    with pytest.raises(ValueError, match='initial'):
        audit.analyze_command_differences(rows(np.zeros((12, 7))), None)
    trace = rows(np.zeros((12, 7)))
    trace[3]['time_s'] += .01
    with pytest.raises(ValueError, match='timestamp'):
        audit.analyze_command_differences(trace, np.zeros(7))
    trace = rows(np.zeros((12, 7)))
    trace[3]['q_command'][1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        audit.analyze_command_differences(trace, np.zeros(7))


def reports():
    legacy = {'hard_failures': [], 'provenance_status': 'passed',
              'legacy_engineering_status': 'passed',
              'worker_lifecycle': {'status': 'passed', 'runner_accepted': True,
                                   'child_returncode': 0, 'verified_from_bound_files': True},
              'motion': {'preference_events': []},
              'contact': {'loaded_drag': {'status': 'within_draft_pending_user_choice',
                                         'events': [], 'max_event_distance_m': 0.,
                                         'max_speed_m_s': 0.}}}
    free = {'status': 'passed', 'failures': [], 'pending': [],
            'limits': {'velocity_rad_s': .25, 'acceleration_rad_s2': 1., 'jerk_rad_s3': 1.5}}
    forward = {'status': 'passed', 'failures': []}
    outcome = {'success': True, 'requested_points': ['L4'],
               'points': [{'target': 'L4', 'success': True}],
               'dataset_role': 'expert_demonstration', 'expert_demonstration': True,
               'executed_controller_class': 'sim.auscultation_skill_v1.controller.ExpertController'}
    return legacy, free, forward, outcome


def test_pilot_can_pass_numerics_but_cannot_self_certify_training_or_external_acceptance():
    result = quality().assess_reports(*reports())
    assert result['episode_simulation_gates_passed']
    assert not result['repetition_gate_passed']
    assert not result['sim_control_imitation_eligible']
    assert result['accepted'] is False
    assert result['external_acceptance'] == 'pending_not_evaluated'


def test_global_command_preference_exceedance_fails_despite_draft_status():
    args = list(reports())
    args[0]['motion']['preference_events'] = [
        {'label': 'commanded', 'reason': 'joint_jerk_rad_s3', 'value': 836., 'limit': 20.,
         'status': 'exceeded_pending_user_choice'}]
    result = quality().assess_reports(*args)
    assert not result['command_quality_passed']


def test_measured_raw_exceedance_remains_visible_without_impersonating_command_failure():
    args = list(reports())
    args[0]['motion']['preference_events'] = [
        {'label': 'measured', 'reason': 'joint_jerk_rad_s3', 'value': 35., 'limit': 20.}]
    result = quality().assess_reports(*args)
    assert result['command_quality_passed']
    assert result['measured_raw_target_status'] == 'exceeded'
    assert result['research_limitations']


@pytest.mark.parametrize('event', [
    {'category': 'retraction', 'distance_m': .0005001, 'max_speed_m_s': .001},
    {'category': 'point_change', 'distance_m': 0., 'max_speed_m_s': 0.},
    {'category': 'free_transfer', 'distance_m': 0., 'max_speed_m_s': 0.}])
def test_loaded_events_fail_by_raw_distance_or_forbidden_category(event):
    args = list(reports())
    args[0]['contact']['loaded_drag']['events'] = [event]
    assert not quality().assess_reports(*args)['loaded_contact_quality_passed']


def test_pending_or_weakened_free_audit_never_passes():
    args = list(reports())
    args[1]['limits']['jerk_rad_s3'] = 20.
    assert not quality().assess_reports(*args)['command_quality_passed']
    args = list(reports())
    args[1]['status'] = 'pending'
    assert not quality().assess_reports(*args)['command_quality_passed']


def test_recovery_and_student_episode_never_become_demonstrations():
    args = list(reports())
    args[3]['dataset_role'] = 'student_evaluation'
    args[3]['expert_demonstration'] = False
    result = quality().assess_reports(*args, repetition_report={'status': 'passed'})
    assert not result['sim_control_imitation_eligible']
    assert not result['learned_task_success']


def test_isolated_legacy_adapter_uses_variable_block_binding_without_mutating_parent():
    from sim.auscultation_expert_layout_v1 import dataset as original_dataset
    from sim.auscultation_skill_v1.trace_binding import validate_control_trace
    before = original_dataset.validate_control_trace
    core = quality()._legacy_equations()
    _, adapter, _, _ = core._legacy_modules()
    assert adapter.validate_control_trace is validate_control_trace
    assert adapter.__file__.endswith('trace_binding.py')
    assert original_dataset.validate_control_trace is before


def test_repetition_rejects_self_asserted_unbound_quality(tmp_path):
    audit = quality()
    report = tmp_path / 'quality.json'
    report.write_text(json.dumps({'schema': 'auscultation_skill_quality_v1',
        'episode': str(tmp_path), 'artifact_sha256': {},
        'contract_sha256': audit.sha(audit.ROOT / 'contract.json'),
        'audit_sources': {str(audit.ROOT / 'quality.py'): audit.sha(audit.ROOT / 'quality.py')},
        'episode_simulation_gates_passed': True,
        'requested_points': list(audit.TARGETS), 'dataset_role': 'expert_demonstration'}))
    with pytest.raises(ValueError, match='provenance'):
        audit._read_bound_quality(report)


def test_repetition_requires_three_bound_episodes_before_final_output(tmp_path):
    audit = quality()
    source = tmp_path / 'qualification.json'
    source.write_text(json.dumps({'schema': 'auscultation_skill_repetition_v1', 'episodes': []}))
    with pytest.raises(ValueError, match='insufficient'):
        audit.verify_repetitions(source)
    output = tmp_path / 'release.json'
    with pytest.raises(ValueError, match='insufficient'):
        audit.qualify_repetitions([], output)
    assert not output.exists()


def test_student_identity_cannot_be_relabelled_as_expert_demonstration():
    args = list(reports())
    args[3]['executed_controller_class'] = 'sim.auscultation_skill_v1.student.StudentController'
    result = quality().assess_reports(*args, repetition_report={
        'status': 'passed', 'verified_from_bound_files': True})
    assert not result['sim_control_imitation_eligible']


@pytest.mark.parametrize('change', [None, {'status': 'failed'}, {'runner_accepted': False},
                                  {'child_returncode': 1}, {'child_returncode': -11},
                                  {'verified_from_bound_files': False}])
def test_saved_success_cannot_override_failed_or_missing_worker_completion(change):
    args = list(reports())
    if change is None:
        del args[0]['worker_lifecycle']
    else:
        args[0]['worker_lifecycle'].update(change)
    result = quality().assess_reports(*args, repetition_report={
        'status': 'passed', 'verified_from_bound_files': True})
    assert not result['hard_simulation_gates_passed']
    assert not result['episode_simulation_gates_passed']
    assert not result['sim_control_imitation_eligible']
    assert not result['assisted_system_success']


def lifecycle_fixture(directory, code):
    audit = quality()
    from sim.auscultation_skill_v1.run import verify_worker_completion
    (directory / 'input_snapshot').mkdir()
    snapshot = directory / 'input_snapshot/manifest.json'
    snapshot.write_text('{}')
    outcome = directory / 'outcome.json'
    outcome.write_text(json.dumps({'reset_id': 'fresh-test-reset',
                                  'input_snapshot_manifest_sha256': audit.sha(snapshot)}))
    completion = directory / 'worker_completion.json'
    completion.write_text(json.dumps({'schema': 'auscultation_skill_worker_completion_v1',
        'returncode': code, 'reset_id': 'fresh-test-reset', 'outcome_sha256': audit.sha(outcome),
        'close_returned': True, 'completion_stage': 'post_app_close_pre_process_exit',
        'exit_policy': 'controlled_os_exit_after_close',
        'input_snapshot_manifest_sha256': audit.sha(snapshot)}))
    lifecycle = verify_worker_completion(directory, code)
    runner = {'accepted': code == 0, 'child_returncode': code, 'worker_lifecycle': lifecycle,
              'evidence_sha256': lifecycle['evidence_sha256']}
    (directory / 'runner_summary.json').write_text(json.dumps(runner))
    return runner


@pytest.mark.parametrize('code', [0, 1])
def test_runner_binding_preserves_normal_failed_return_without_marking_success(tmp_path, code):
    lifecycle_fixture(tmp_path, code)
    result = quality().verify_runner_lifecycle(tmp_path)
    assert result['status'] == 'passed'
    assert result['runner_accepted'] is (code == 0)
    assert result['child_returncode'] == code
    assert set(result['artifact_sha256']) == {'runner_summary.json', 'worker_completion.json'}


def test_runner_binding_rejects_parent_receipt_hash_mutation(tmp_path):
    runner = lifecycle_fixture(tmp_path, 0)
    runner['evidence_sha256']['worker_completion.json'] = '0' * 64
    (tmp_path / 'runner_summary.json').write_text(json.dumps(runner))
    with pytest.raises(ValueError, match='lifecycle|binding'):
        quality().verify_runner_lifecycle(tmp_path)


def test_runner_binding_rejects_completion_from_another_outcome(tmp_path):
    lifecycle_fixture(tmp_path, 0)
    outcome = tmp_path / 'outcome.json'
    data = json.loads(outcome.read_text()); data['reset_id'] = 'different-reset'
    outcome.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='worker completion'):
        quality().verify_runner_lifecycle(tmp_path)


def captured_repetition_fixture(tmp_path, monkeypatch):
    """Mock only completed physics qualification; source objects are real bytes."""
    from sim.auscultation_skill_v1 import dataset
    audit = quality()
    # Isolate a current deployment so tests change actual source bytes without
    # modifying the shared workspace or weakening inventory verification.
    deployment = tmp_path / 'deployment'
    deployment.mkdir()
    for name in dataset.EXECUTOR_CODE_FILES:
        (deployment / name).write_bytes((audit.ROOT / name).read_bytes())
    monkeypatch.setattr(dataset, '__file__', str(deployment / 'dataset.py'))
    monkeypatch.setattr(audit, '_read_bound_quality', audit.read)
    quality_paths = []
    for index in range(3):
        episode = tmp_path / ('nominal_' + str(index))
        objects = episode / 'input_snapshot/objects'
        objects.mkdir(parents=True)
        files, artifacts = {}, {'expert_trace.jsonl': str(index) * 64}
        for name in dataset.EXECUTOR_CODE_FILES:
            content = (deployment / name).read_bytes()
            digest = dataset.sha(content)
            (objects / digest).write_bytes(content)
            files['source/contact_skill/' + name] = {'sha256': digest}
            artifacts['input_snapshot/objects/' + digest] = digest
        snapshot = episode / 'input_snapshot/manifest.json'
        snapshot.write_text(json.dumps({'files': files}))
        artifacts['input_snapshot/manifest.json'] = audit.sha(snapshot)
        (episode / 'outcome.json').write_text(json.dumps({
            'reset_id': 'reset-' + str(index), 'condition_id': 'nominal_qualification', 'speed_scale': 1.,
            'input_snapshot': 'input_snapshot/manifest.json',
            'input_snapshot_manifest_sha256': audit.sha(snapshot)}))
        path = tmp_path / ('nominal_' + str(index) + '.quality.json')
        path.write_text(json.dumps({'episode': str(episode), 'episode_simulation_gates_passed': True,
            'requested_points': list(audit.TARGETS), 'dataset_role': 'expert_demonstration',
            'artifact_sha256': artifacts}))
        quality_paths.append(path)
    return audit, dataset, deployment, quality_paths


def test_repetition_report_saves_fresh_captured_executor_inventory(tmp_path, monkeypatch):
    audit, dataset, _, paths = captured_repetition_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'repetitions.json'
    report = audit.qualify_repetitions(paths, output)
    expected = dataset.current_executor_inventory()
    assert report['executor_code_sha256'] == expected
    assert audit.read(output)['executor_code_sha256'] == expected
    assert audit.verify_repetitions(output)['executor_code_sha256'] == expected
    report['executor_code_sha256']['controller.py'] = '0' * 64
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='repetition.*executor'):
        audit.verify_repetitions(output)


@pytest.mark.parametrize('filename', ['controller.py', 'curve.py', 'executor.py',
                                     'config.py', 'contract.py', 'observations.py'])
def test_actual_executor_source_drift_rejects_old_repetitions_before_data_eligibility(tmp_path, monkeypatch, filename):
    audit, dataset, deployment, paths = captured_repetition_fixture(tmp_path, monkeypatch)
    repetitions = tmp_path / 'repetitions.json'
    audit.qualify_repetitions(paths, repetitions)
    episode = tmp_path / 'collection'
    episode.mkdir()
    report = tmp_path / 'collection.quality.json'
    report.write_text(json.dumps({'episode': str(episode), 'sim_control_imitation_eligible': True,
        'repetition_report': {'path': str(repetitions), 'sha256': audit.sha(repetitions)}}))
    entry = {'episode_dir': str(episode), 'quality_report': str(report)}
    assert dataset.verify_eligibility(entry)['sim_control_imitation_eligible']
    # Contract JSON, stored inventories, quality files and repetition hashes all
    # stay identical: only current controller/curve/etc source content changes.
    source = deployment / filename
    source.write_bytes(source.read_bytes() + b'\n# changed executable source revision\n')
    with pytest.raises(ValueError, match='executor source'):
        dataset.verify_eligibility(entry)


def test_repetitions_reject_changed_captured_source_object(tmp_path, monkeypatch):
    audit, _, _, paths = captured_repetition_fixture(tmp_path, monkeypatch)
    repetitions = tmp_path / 'repetitions.json'
    audit.qualify_repetitions(paths, repetitions)
    episode = tmp_path / 'nominal_1'
    snapshot = audit.read(episode / 'input_snapshot/manifest.json')
    digest = snapshot['files']['source/contact_skill/curve.py']['sha256']
    source = episode / 'input_snapshot/objects' / digest
    source.write_bytes(source.read_bytes() + b'\n# captured object changed\n')
    with pytest.raises(ValueError, match='source object'):
        audit.verify_repetitions(repetitions)


def test_repetitions_compare_current_code_even_when_old_report_has_no_inventory(tmp_path, monkeypatch):
    audit, _, deployment, paths = captured_repetition_fixture(tmp_path, monkeypatch)
    repetitions = tmp_path / 'repetitions.json'
    report = audit.qualify_repetitions(paths, repetitions)
    report.pop('executor_code_sha256', None)
    repetitions.write_text(json.dumps(report))
    source = deployment / 'controller.py'
    source.write_bytes(source.read_bytes() + b'\n# new controller without config change\n')
    with pytest.raises(ValueError, match='executor source'):
        audit.verify_repetitions(repetitions)
