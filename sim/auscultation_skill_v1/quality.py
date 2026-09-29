"""Independent simulation-only eligibility; inherited reports remain unchanged."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .config import ROOT, TARGETS, PHYSICS_DT_S, ENGINEERING_GATES as GATES
from .contract import read, sha, require, verify_contract, validate_outcome_contract


def _legacy_equations():
    from sim.expert_quality_20260924 import quality as original
    from .provenance import verify_candidate_sources
    spec = importlib.util.spec_from_file_location('wam_skill_v1_legacy_quality', original.__file__)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.verify_candidate_sources = verify_candidate_sources
    original_modules = core._legacy_modules
    def bound_modules():
        from types import SimpleNamespace
        from . import trace_binding
        posture, _, controller, kinematics = original_modules()
        adapter = SimpleNamespace(validate_control_trace=trace_binding.validate_control_trace,
                                  __file__=trace_binding.__file__)
        return posture, adapter, controller, kinematics
    core._legacy_modules = bound_modules
    return core


def validate_episode_provenance(episode):
    return _legacy_equations().validate_episode_provenance(episode)


def verify_runner_lifecycle(episode):
    """Bind the parent result and the receipt written after Kit closes normally.

    A normal return with code 1 is valid failed-run evidence, useful for negative
    controls. It never passes the episode's hard simulation gate.
    """
    from .run import verify_worker_completion
    source = Path(episode)
    runner_path = source / 'runner_summary.json'
    expected_runner = sha(runner_path)
    runner = read(runner_path)
    code = runner.get('child_returncode')
    actual = verify_worker_completion(source, code)
    require(actual['status'] == 'passed', 'worker completion rejected: ' + str(actual['reasons']))
    require(runner.get('worker_lifecycle') == actual, 'parent worker lifecycle differs from fresh verification')
    require(type(runner.get('accepted')) is bool, 'parent runner acceptance is missing')
    for name, digest in actual['evidence_sha256'].items():
        require(runner.get('evidence_sha256', {}).get(name) == digest, 'parent completion hash binding changed')
    require(sha(runner_path) == expected_runner, 'runner changed during verification')
    return dict(actual, runner_accepted=runner['accepted'], child_returncode=code,
                verified_from_bound_files=True,
                artifact_sha256=dict(actual['evidence_sha256'], **{'runner_summary.json': expected_runner}))


def analyze_command_differences(rows, initial_q):
    """Every real forward window, including mixed phases and row-0 predecessor.

    No invented stationary padding, filtering, unwrap, or phase exclusions.
    Index -1 is the explicitly recorded retained command before physics row 0.
    """
    require(isinstance(rows, list) and len(rows) >= 4, 'insufficient command samples')
    require(initial_q is not None, 'initial retained command is required')
    initial = np.asarray(initial_q, float)
    commands = np.asarray([row['q_command'] for row in rows], float)
    times = np.asarray([row['time_s'] for row in rows], float)
    n = len(rows)
    require(initial.shape == (7,) and commands.shape == (n, 7)
            and np.isfinite(initial).all() and np.isfinite(commands).all(), 'nonfinite or malformed command')
    require(times.shape == (n,) and np.isfinite(times).all()
            and np.allclose(times, np.arange(1, n + 1) * PHYSICS_DT_S, atol=1e-7, rtol=0),
            'timestamp gap or origin changed')
    labels = [tuple(row[key] for key in ('target', 'phase', 'subphase')) for row in rows]
    require(all(all(isinstance(value, str) and value for value in label) for label in labels),
            'invalid command phase labels')
    values = np.vstack((initial, commands))
    labels = [labels[0]] + labels
    report = {'schema': 'skill_all_command_forward_differences_v1', 'status': 'passed',
              'sample_count': n, 'dt_s': PHYSICS_DT_S, 'initial_retained_command_available': True,
              'excluded_window_count': 0, 'metrics': {}, 'failures': [],
              'method': 'unfiltered forward differences across every saved command and all phase boundaries'}
    for metric, order, gate in (('velocity_rad_s', 1, 'joint_velocity_rad_s'),
                                ('acceleration_rad_s2', 2, 'joint_acceleration_rad_s2'),
                                ('jerk_rad_s3', 3, 'joint_jerk_rad_s3')):
        differences = np.diff(values, n=order, axis=0) / PHYSICS_DT_S ** order
        absolute = np.abs(differences)
        mixed = np.asarray([len(set(labels[i:i + order + 1])) > 1
                            for i in range(len(differences))], bool)
        exceeded = np.any(absolute > GATES[gate] + 1e-9, axis=1)
        start, joint = np.unravel_index(absolute.argmax(), absolute.shape)
        peak = {'value': float(absolute[start, joint]), 'signed_value': float(differences[start, joint]),
                'joint': int(joint + 1), 'start_row': int(start - 1),
                'end_row': int(start + order - 1), 'time_s': float((start + order) * PHYSICS_DT_S),
                'mixed_boundary': bool(mixed[start])}
        report['metrics'][metric] = {'limit': GATES[gate], 'order': order,
            'evaluated_windows': len(differences), 'violation_windows': int(exceeded.sum()),
            'mixed_boundary_windows': int(mixed.sum()),
            'mixed_boundary_violation_windows': int((mixed & exceeded).sum()),
            'max_abs_per_joint': absolute.max(axis=0).tolist(), 'peak': peak}
        if exceeded.any():
            report['failures'].append(metric + '_limit')
    if report['failures']:
        report['status'] = 'failed'
    return report


def assess_reports(legacy, free, command_forward, outcome, repetition_report=None):
    """Pure gate composition; file auditor verifies all supplied evidence first."""
    failures = []
    hard = (legacy.get('hard_failures') == [] and legacy.get('provenance_status') == 'passed'
            and legacy.get('legacy_engineering_status') == 'passed')
    lifecycle = legacy.get('worker_lifecycle', {})
    process_passed = (lifecycle.get('status') == 'passed'
                      and lifecycle.get('runner_accepted') is True
                      and type(lifecycle.get('child_returncode')) is int
                      and lifecycle['child_returncode'] == 0
                      and lifecycle.get('verified_from_bound_files') is True)
    points = outcome.get('requested_points')
    complete = (isinstance(points, list) and bool(points) and len(points) == len(set(points))
                and all(point in TARGETS for point in points) and outcome.get('success') is True
                and [p.get('target') for p in outcome.get('points', [])] == points
                and all(p.get('success') is True for p in outcome.get('points', [])))
    hard = bool(hard and complete and process_passed)
    if not hard:
        failures.append('hard_simulation_or_completion_gate')
    preferences = legacy.get('motion', {}).get('preference_events')
    known_preferences = isinstance(preferences, list)
    preferences = preferences if known_preferences else []
    command_events = [event for event in preferences if event.get('label') == 'commanded']
    measured_events = [event for event in preferences if event.get('label') == 'measured']
    exact_free_limits = {'velocity_rad_s': GATES['free_velocity_rad_s'],
                         'acceleration_rad_s2': GATES['free_acceleration_rad_s2'],
                         'jerk_rad_s3': GATES['free_jerk_rad_s3']}
    free_pass = (free.get('status') == 'passed' and free.get('failures') == []
                 and free.get('pending') == [] and free.get('limits') == exact_free_limits)
    command = bool(known_preferences and not command_events and free_pass
                   and command_forward.get('status') == 'passed' and command_forward.get('failures') == [])
    if not command:
        failures.append('command_quality_gate')
    drag = legacy.get('contact', {}).get('loaded_drag', {})
    events = drag.get('events')
    loaded = (drag.get('status') in ('passed', 'failed', 'within_draft_pending_user_choice',
                                     'exceeded_pending_user_choice') and isinstance(events, list))
    loaded_failures = []
    if loaded:
        for event in events:
            category = event.get('category')
            distance = event.get('tangential_distance_m', event.get('distance_m'))
            speed = event.get('max_speed_m_s')
            if (category not in ('retraction', 'point_change', 'free_transfer')
                    or type(distance) not in (int, float) or type(speed) not in (int, float)
                    or not np.isfinite([distance, speed]).all() or min(distance, speed) < 0):
                loaded_failures.append('malformed_loaded_event')
            elif category != 'retraction':
                loaded_failures.append('loaded_' + category)
            elif distance > GATES['loaded_drag_max_event_m'] or speed > GATES['loaded_drag_max_speed_m_s']:
                loaded_failures.append('loaded_retraction_tangent_limit')
        for field, gate in (('max_event_distance_m', 'loaded_drag_max_event_m'),
                            ('max_speed_m_s', 'loaded_drag_max_speed_m_s')):
            value = drag.get(field)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= GATES[gate]:
                loaded_failures.append(field + '_limit_or_missing')
    loaded = bool(loaded and not loaded_failures)
    if not loaded:
        failures.append('loaded_contact_quality_gate')
    repetition = bool(repetition_report and repetition_report.get('status') == 'passed'
                      and repetition_report.get('verified_from_bound_files') is True)
    expert = (outcome.get('dataset_role') == 'expert_demonstration'
              and outcome.get('expert_demonstration') is True
              and outcome.get('executed_controller_class') == 'sim.auscultation_skill_v1.controller.ExpertController')
    clean = not any(outcome.get(name) for name in ('recovery_used', 'expert_fallback_used', 'failure_latched'))
    episode_pass = hard and command and loaded and clean
    limitations = ['Measured raw derivative targets remain diagnostic; physical, anatomical and acoustic acceptance is pending.']
    if measured_events:
        limitations.append('Measured raw engineering targets exceeded; retained in legacy report and never declared achieved.')
    return {'accepted': False, 'external_acceptance': 'pending_not_evaluated',
            'worker_lifecycle': lifecycle, 'worker_process_gate_passed': process_passed,
            'hard_simulation_gates_passed': hard, 'command_quality_passed': command,
            'loaded_contact_quality_passed': loaded, 'repetition_gate_passed': repetition,
            'episode_simulation_gates_passed': episode_pass,
            'sim_control_imitation_eligible': bool(episode_pass and repetition and expert),
            'learned_task_success': False,
            'assisted_system_success': bool(complete and hard),
            'learned_success_scope': 'Not established by expert quality audit; requires independent action-attribution evaluation.',
            'measured_raw_target_status': 'exceeded' if measured_events else 'within_bounds' if known_preferences else 'unavailable',
            'measured_raw_target_events': measured_events, 'command_preference_events': command_events,
            'loaded_contact_failures': sorted(set(loaded_failures)), 'failures': failures,
            'research_limitations': limitations}


def _read_bound_quality(path):
    """Reports are evidence only while every recorded input and audit source matches."""
    path = Path(path).resolve()
    report = read(path)
    require(report.get('schema') == 'auscultation_skill_quality_v1', 'wrong quality report schema')
    source = Path(report['episode']).resolve()
    require(source.is_dir(), 'quality episode is unavailable')
    core = _legacy_equations()
    provenance = core.validate_episode_provenance(source)
    require(provenance.get('status') == 'passed', 'quality episode provenance is not valid')
    required = set(core.REQUIRED_ARTIFACTS)
    require(required.issubset(report.get('artifact_sha256', {})), 'quality provenance bindings missing')
    require(not core.verify_hashes(source, report['artifact_sha256']), 'quality input artifact changed')
    require({'runner_summary.json', 'worker_completion.json'}.issubset(report['artifact_sha256']),
            'quality worker lifecycle bindings missing')
    lifecycle = verify_runner_lifecycle(source)
    require(report.get('worker_lifecycle') == lifecycle, 'quality worker lifecycle changed')
    if report.get('hard_simulation_gates_passed') is True:
        require(lifecycle['runner_accepted'] is True and lifecycle['child_returncode'] == 0,
                'quality accepted an unsuccessful worker')
    require(report.get('contract_sha256') == sha(ROOT / 'contract.json'), 'quality contract changed')
    for filename, expected in report.get('audit_sources', {}).items():
        require(sha(filename) == expected, 'quality evaluator changed')
    require(bool(report.get('audit_sources')), 'quality evaluator bindings missing')
    inherited = report.get('inherited_reports', {})
    require(set(inherited) == {'legacy', 'free_motion'}, 'inherited quality evidence missing')
    for item in inherited.values():
        require(sha(item['path']) == item['sha256'], 'inherited quality report changed')
    return report


def verify_repetitions(path):
    # This helper reads already-bound snapshot objects only; it does not call
    # dataset eligibility or recurse into repetition verification.
    from .dataset import captured_executor_inventory, verify_executor_inventory
    report = read(path)
    require(report.get('schema') == 'auscultation_skill_repetition_v1', 'wrong repetition schema')
    entries = report.get('episodes')
    require(isinstance(entries, list) and len(entries) >= GATES['minimum_nominal_repetitions'],
            'insufficient nominal repetitions')
    resets, episodes, speeds, trace_hashes = set(), set(), set(), []
    executor_inventory, captured_sources = None, {}
    for entry in entries:
        quality_path = Path(entry['path'])
        require(sha(quality_path) == entry['sha256'], 'repetition quality report changed')
        quality = _read_bound_quality(quality_path)
        require(quality.get('episode_simulation_gates_passed') is True
                and quality.get('requested_points') == list(TARGETS)
                and quality.get('dataset_role') == 'expert_demonstration', 'repetition episode did not qualify')
        outcome = read(Path(quality['episode']) / 'outcome.json')
        inventory, source_hashes = captured_executor_inventory(quality['episode'], outcome, quality)
        if executor_inventory is None:
            executor_inventory = inventory
        require(inventory == executor_inventory, 'nominal repetitions used different executor sources')
        captured_sources.update(source_hashes)
        reset = outcome.get('reset_id')
        require(isinstance(reset, str) and bool(reset) and reset not in resets, 'duplicate or missing fresh reset identity')
        require(outcome.get('condition_id') == 'nominal_qualification', 'repetition must be a nominal qualification run')
        require(quality['episode'] not in episodes, 'duplicate repetition episode')
        resets.add(reset); episodes.add(quality['episode']); speeds.add(outcome['speed_scale'])
        trace_hashes.append(quality['artifact_sha256']['expert_trace.jsonl'])
    require(len(speeds) == 1, 'nominal repetition speeds differ')
    require(report.get('contract_sha256') == sha(ROOT / 'contract.json'), 'repetition contract changed')
    if 'executor_code_sha256' in report:
        require(report['executor_code_sha256'] == executor_inventory, 'repetition executor inventory changed')
    # Recheck both live code and captured objects after the complete inventory
    # has been examined; a source change during verification cannot pass.
    verify_executor_inventory(executor_inventory)
    for source, expected in captured_sources.items():
        require(sha(source) == expected, 'repetition captured executor source changed')
    return dict(report, status='passed', verified_from_bound_files=True,
                executor_code_sha256=executor_inventory,
                distinct_reset_count=len(resets), distinct_trace_count=len(set(trace_hashes)),
                scope='Fresh worker resets and captured executor sources equal the current executor; deterministic repeated traces are not training variation.')


def qualify_repetitions(report_paths, output):
    output = Path(output).resolve()
    require(not output.exists(), 'repetition output must be new')
    report = {'schema': 'auscultation_skill_repetition_v1', 'status': 'pending',
              'contract_sha256': sha(ROOT / 'contract.json'),
              'episodes': [{'path': str(Path(path).resolve()), 'sha256': sha(path)} for path in report_paths]}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate in memory through the same reader using a temporary sibling. The
    # final file is written only after all evidence is checked.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='skill-repetitions-', dir=output.parent) as temporary:
        probe = Path(temporary) / 'repetitions.json'
        probe.write_text(json.dumps(report, allow_nan=False))
        report = verify_repetitions(probe)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def audit_episode(episode, output, repetition_report=None, *, contract_path=ROOT / 'contract.json'):
    """Recompute legacy, free and full command gates without changing episode bytes."""
    source, destination = Path(episode).resolve(), Path(output).resolve()
    require(destination != source and source not in destination.parents, 'audit output must be outside episode')
    require(not destination.exists(), 'quality output must be new')
    paths = {name: destination.with_name(destination.stem + '.' + name + '.json')
             for name in ('legacy', 'free_motion')}
    require(not any(path.exists() for path in paths.values()), 'derived quality outputs must be new')
    source_paths = [Path(__file__), *(ROOT / name for name in
        ('config.py', 'contract.py', 'provenance.py', 'trace_binding.py', 'audits.py', 'run.py')),
        ROOT.parent / 'expert_quality_20260924/quality_profile.json']
    audit_hashes = {str(path.resolve()): sha(path) for path in source_paths}
    core = _legacy_equations()
    contract = verify_contract(contract_path)
    # The inherited profile must retain exactly the numerical targets we adopted.
    profile = core.load_profile()
    for key in ('joint_step_rad', 'joint_velocity_rad_s', 'joint_acceleration_rad_s2', 'joint_jerk_rad_s3',
                'loaded_force_N', 'loaded_drag_max_event_m', 'loaded_drag_max_speed_m_s', 'filter_window_s'):
        require(profile[key] == GATES[key], 'inherited quality profile differs from frozen V1 goals')
    legacy = core.audit_episode(source, paths['legacy'])
    from sim.expert_quality_20260924 import audit_free_motion as old_free
    spec = importlib.util.spec_from_file_location('sim.expert_quality_20260924._skill_free_quality', old_free.__file__)
    free_core = importlib.util.module_from_spec(spec); spec.loader.exec_module(free_core)
    free_core.validate_episode_provenance = core.validate_episode_provenance
    free = free_core.audit_free_motion(source, paths['free_motion'])
    for item in legacy.get('audit_sources', {}).values():
        if isinstance(item, dict) and 'path' in item and 'sha256' in item:
            audit_hashes[str(Path(item['path']).resolve())] = item['sha256']
    for module_path in (Path(core.__file__), Path(old_free.__file__)):
        audit_hashes.setdefault(str(module_path.resolve()), sha(module_path))
    outcome = read(source / 'outcome.json')
    forward = {'status': 'failed', 'failures': ['command_evidence_unavailable']}
    extra_failures = []
    lifecycle_hashes = {name: sha(source / name) for name in
                        ('runner_summary.json', 'worker_completion.json') if (source / name).is_file()}
    try:
        lifecycle = verify_runner_lifecycle(source)
    except (OSError, ValueError, KeyError, TypeError) as error:
        lifecycle = {'status': 'failed', 'verified_from_bound_files': False, 'reasons': [str(error)]}
        extra_failures.append('worker_lifecycle:' + str(error))
    legacy = dict(legacy, worker_lifecycle=lifecycle,
                  artifact_sha256=dict(legacy.get('artifact_sha256', {}), **lifecycle_hashes))
    try:
        validate_outcome_contract(outcome, contract, outcome.get('requested_points'))
        rows = [json.loads(line) for line in (source / 'expert_trace.jsonl').read_text().splitlines()]
        controls = read(source / 'control_commands.json')
        require(isinstance(controls, list) and bool(controls) and controls[0].get('physics_row') == 0,
                'initial retained command binding is missing')
        forward = analyze_command_differences(rows, controls[0].get('previous_q_command'))
        if any(row.get('recovery') is not False or row.get('hard_clamp_intervened') is True or row.get('failure') is True
               or row.get('expert_fallback_used') is True for row in controls):
            extra_failures.append('recovery_or_undeclared_action_intervention')
        manifest = read(source / outcome['input_snapshot'])
        for point in outcome['requested_points']:
            require(manifest['files'][point + '/plan']['sha256'] == contract['selected_plan_sha256'][point],
                    'executed plan differs from accepted posture: ' + point)
        require(manifest['files']['source/contact_skill/run_contract.json']['sha256'] == sha(contract_path),
                'executed numerical contract differs')
    except (OSError, ValueError, TypeError, KeyError) as error:
        extra_failures.append('skill_contract_or_command_binding:' + str(error))
    repetitions = None
    if repetition_report is not None:
        try:
            repetitions = verify_repetitions(repetition_report)
        except (OSError, ValueError, TypeError, KeyError) as error:
            extra_failures.append('repetition_binding:' + str(error))
    # Any source/input mutation invalidates every downstream eligibility claim.
    extra_failures += core.verify_hashes(source, legacy.get('artifact_sha256', {}))
    for path, expected in audit_hashes.items():
        if sha(path) != expected:
            extra_failures.append('audit_source_changed:' + path)
    if extra_failures:
        legacy = dict(legacy, hard_failures=list(legacy.get('hard_failures', [])) + extra_failures,
                      legacy_engineering_status='failed')
    result = assess_reports(legacy, free, forward, outcome, repetitions)
    result.update(schema='auscultation_skill_quality_v1', episode=str(source),
        requested_points=outcome.get('requested_points'), dataset_role=outcome.get('dataset_role'),
        controller_type=outcome.get('controller_type'), reset_id=outcome.get('reset_id'),
        speed_scale=outcome.get('speed_scale'), condition_id=outcome.get('condition_id'),
        artifact_sha256=legacy.get('artifact_sha256', {}), contract_sha256=sha(contract_path),
        audit_sources=audit_hashes, command_forward=forward, hard_failures=legacy.get('hard_failures', []),
        inherited_reports={name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()},
        repetition_report=({'path': str(Path(repetition_report).resolve()), 'sha256': sha(repetition_report)}
                           if repetition_report is not None else None),
        scope='Versioned simulation-control engineering eligibility only; inherited acoustic/clinical/assembly acceptance stays pending.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', nargs='?', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetition-report', type=Path)
    parser.add_argument('--qualify-repetitions', nargs='+', type=Path)
    args = parser.parse_args()
    if args.qualify_repetitions:
        result = qualify_repetitions(args.qualify_repetitions, args.output)
        print(json.dumps({'status': result['status'], 'repetitions': result['distinct_reset_count']}))
        return 0
    if args.episode is None:
        parser.error('episode is required')
    result = audit_episode(args.episode, args.output, args.repetition_report)
    print(json.dumps({key: result[key] for key in ('episode_simulation_gates_passed',
        'sim_control_imitation_eligible', 'measured_raw_target_status', 'hard_failures', 'failures')}))
    return 0 if result['episode_simulation_gates_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
