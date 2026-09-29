"""Synthetic failure paths only; these fixtures are never acceptance receipts."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.auscultation_pi05_v9 import trigger as T


def rows(n=120):
    return [dict(phase='FREE', start=i % 15, episode_id='synthetic_' + str(i // 15)) for i in range(n)]


def bounds(joint=3, value=-.001, *, fixed=False, empty=False):
    lower = [-1.] * 7; lower[3] = 0.
    upper = [1.] * 7; upper[3] = 2.443461
    violation = (dict(row=0, horizon=0, joint=joint, value_rad=value,
                      lower_rad=lower[joint], upper_rad=upper[joint]) if fixed else
                 dict(row_id=0, horizon_index=0, joint_index=joint, proposed_rad=value))
    return dict(coordinate_count=(160 if fixed else 120) * 70, tolerance_rad=0.,
                all_passed=empty, lower_rad=lower, upper_rad=upper,
                violations=[] if empty else [violation],
                **{('violation_count' if fixed else 'violating_coordinates'): 0 if empty else 1})


def selected_fixture():
    candidates = []
    for step in T.STEPS:
        c = dict(local_step=step, optimizer_step=step, eligible=False, deploy_candidate_eligible=False,
                 bounds=bounds(), moving_buckets={'passed': 16 if step == 4867 else 15},
                 reasons=['raw_absolute_urdf_violation'] if step == 4867 else
                         ['raw_absolute_urdf_violation', 'moving_bucket_failed_or_unassessable'])
        candidates.append(c)
    return dict(mode='B_horizon_free_approach', full_budget_complete=True,
                candidate_count=13, candidate_steps=list(T.STEPS), candidates=candidates,
                selected=None, status='rejected_no_eligible_candidate'), {step: {'rows': rows()} for step in T.STEPS}


def test_negative_selection_requires_a_positive_motion_capable_q4_witness():
    selection, reports = selected_fixture()
    result = T.classify_selection(selection, reports)
    assert result['negative_selection_condition'] is True
    assert result['q4_only_motion_capable_steps'] == [4867]
    assert len(result['candidate_table']) == 13
    selection['candidates'][4]['moving_buckets']['passed'] = 15
    selection['candidates'][4]['reasons'].append('moving_bucket_failed_or_unassessable')
    assert not T.classify_selection(selection, reports)['negative_selection_condition']


def test_step_zero_cannot_trigger():
    selection, reports = selected_fixture()
    for c in selection['candidates']:
        c['moving_buckets']['passed'] = 15
    selection['candidates'][0]['moving_buckets']['passed'] = 16
    assert not T.classify_selection(selection, reports)['negative_selection_condition']


@pytest.mark.parametrize('mutation', ['partial_budget', 'missing_candidate', 'missing_report', 'wrong_mode', 'false_eligibility'])
def test_partial_or_inconsistent_selection_rejected(mutation):
    selection, reports = selected_fixture()
    if mutation == 'partial_budget': selection['full_budget_complete'] = False
    elif mutation == 'missing_candidate': selection['candidates'].pop()
    elif mutation == 'missing_report': reports.pop(0)
    elif mutation == 'wrong_mode': selection['mode'] = 'A_pooled_control'
    else: selection['candidates'][4]['eligible'] = True
    with pytest.raises(ValueError): T.classify_selection(selection, reports)


def test_selected_B_is_not_a_negative_selection_trigger():
    selection, reports = selected_fixture()
    c = selection['candidates'][4]; c.update(bounds=bounds(empty=True), reasons=[], eligible=True, deploy_candidate_eligible=True)
    selection.update(selected=c, status='train_fit_candidate_requires_pair_and_fixed160')
    result = T.classify_selection(selection, reports)
    assert result['selected_step'] == 4867 and not result['negative_selection_condition']


@pytest.mark.parametrize('fixed', [False, True])
def test_both_q4_sides_and_other_joint_are_distinguished(fixed):
    n = 160 if fixed else 120
    lower = T.classify_bounds(bounds(fixed=fixed), rows(n), fixed160=fixed)
    upper = T.classify_bounds(bounds(value=2.5, fixed=fixed), rows(n), fixed160=fixed)
    other = T.classify_bounds(bounds(joint=2, value=-2, fixed=fixed), rows(n), fixed160=fixed)
    assert lower['exclusively_q4'] and lower['lower_count'] == 1 and lower['upper_count'] == 0
    assert upper['exclusively_q4'] and upper['upper_count'] == 1 and upper['lower_count'] == 0
    assert not other['exclusively_q4']


@pytest.mark.parametrize('mutation', ['duplicate', 'nonfinite', 'within_bound', 'tolerance', 'denominator', 'missing', 'bool_index'])
def test_malformed_violations_fail_closed(mutation):
    b = bounds()
    if mutation == 'duplicate': b['violations'] *= 2; b['violating_coordinates'] = 2
    elif mutation == 'nonfinite': b['violations'][0]['proposed_rad'] = float('nan')
    elif mutation == 'within_bound': b['violations'][0]['proposed_rad'] = .1
    elif mutation == 'tolerance': b['tolerance_rad'] = 1e-6
    elif mutation == 'denominator': b['coordinate_count'] = 70
    elif mutation == 'missing': b['violations'] = []
    else: b['violations'][0]['row_id'] = False
    with pytest.raises(ValueError): T.classify_bounds(b, rows())


def lifecycle_fixture():
    command = [str(T.BASE / 'vendor/openpi/.venv/bin/python'), '-u', '-B', '-m',
               'sim.auscultation_pi05_v8.train', '--manifest',
               str(T.BASE / 'datasets/pi05_v6_task_pose_armature02_v1/task_conditioned_bound/manifest.json'),
               '--mode', 'B_horizon_free_approach', '--output', str(T.B_RUN),
               '--paired-initialization', str(T.BASE / 'runs/horizon-pair-original8-v1-A/initialization.json')]
    def stage(name, pid, argv):
        return dict(stage=name, command=argv, exit_code=0, passed=True, process_group_drained=True,
                    errors=[], signals=[], deferred_signals=[],
                    process_identity=dict(pid=pid, pgid=pid, sid=pid, start_ticks='12345'))
    stages = [stage('cpu_preflight', 100, command + ['--verify-only']), stage('training', 200, command)]
    training = dict(training_complete=True, local_step=14600, optimizer_step=14600,
                    schedule_cursor=58400, interrupted=False, error=None, pid=200, resume=None)
    prep = {'synthetic': True}
    life = dict(schema='wam_horizon_pair_arm_lifecycle_v1', arm='B', mode='B_horizon_free_approach',
                execute=True, completed=True, training_complete=True, gpu_released=True, errors=[],
                output=str(T.B_RUN), training_receipt=training, preparation=prep,
                preparation_sha256=T.PREPARATION_SHA, stages=stages, command=command)
    return life, training, stages[1], prep


def test_clean_synthetic_lifecycle_classifier():
    T.verify_lifecycle(*lifecycle_fixture())


@pytest.mark.parametrize('mutation', ['incomplete', 'false_exit', 'undrained', 'signal', 'nonowned', 'wrong_command', 'interrupted', 'wrong_clock', 'wrong_pid'])
def test_bad_actual_exit_or_training_cannot_support_condition(mutation):
    life, training, process, prep = lifecycle_fixture()
    if mutation == 'incomplete': life['completed'] = False
    elif mutation == 'false_exit': process['exit_code'] = False
    elif mutation == 'undrained': process['process_group_drained'] = False
    elif mutation == 'signal': process['deferred_signals'] = ['SIGTERM']
    elif mutation == 'nonowned': process['process_identity']['sid'] += 1
    elif mutation == 'wrong_command': process['command'][4] = 'another.module'
    elif mutation == 'interrupted': training['interrupted'] = True
    elif mutation == 'wrong_clock': training['optimizer_step'] = 18599
    else: training['pid'] = 201
    with pytest.raises(ValueError): T.verify_lifecycle(life, training, process, prep)


def test_hash_is_checked_before_json_parse_and_symlink_rejected(tmp_path):
    path = tmp_path / 'invalid.json'; path.write_text('invalid json')
    with pytest.raises(ValueError, match='hash differs'): T.pinned(path, '0' * 64)
    path.write_text('{}'); link = tmp_path / 'link'; link.symlink_to(path)
    with pytest.raises(ValueError, match='Canonical|Symlink'): T.sha(link)


def test_receipt_is_freshly_reconstructed_not_just_self_hashed(tmp_path, monkeypatch):
    saved = dict(schema=T.SCHEMA, condition_satisfied=True, b_selection_lock_binding={'path': '/b', 'sha256': 'b'},
                 lifecycle_binding={'path': '/life', 'sha256': 'l'}, specification_binding={'path': '/spec'},
                 design_review_binding={'path': '/review'}, fixed_plan_binding=None, fixed_receipt_binding=None)
    path = tmp_path / 'receipt.json'; path.write_text(json.dumps(saved)); digest = T.sha(path)
    monkeypatch.setattr(T, 'evaluate', lambda **kwargs: dict(saved, condition_satisfied=False))
    with pytest.raises(ValueError, match='not satisfied'): T.verify_receipt(path, digest)


def full_wiring_fixture(tmp_path, monkeypatch, *, selected=False):
    """Real temporary I/O; existing B scientific verifiers are explicit stubs."""
    base = tmp_path.resolve(); run = base / 'runs/horizon-pair-original8-v1-B'
    monkeypatch.setattr(T, 'BASE', base); monkeypatch.setattr(T, 'B_RUN', run)
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return path
    project = base / 'original_project'
    ef = put(project / 'sim/auscultation_pi05_v8/evaluation.py', {'synthetic': 'E'})
    pf = put(project / 'sim/auscultation_pi05_v8/pair_admission.py', {'synthetic': 'P'})
    monkeypatch.setattr(T, 'EVALUATION_SHA', T.sha(ef)); monkeypatch.setattr(T, 'PAIR_SHA', T.sha(pf))
    specification = put(base / 'spec.json', dict(schema='wam_q4_rational_conditional_experiment_design_v1', current_AB_immutable=True))
    review = put(base / 'review.json', dict(passed=True))
    monkeypatch.setattr(T, 'SPEC_SHA', T.sha(specification)); monkeypatch.setattr(T, 'DESIGN_REVIEW_SHA', T.sha(review))
    guard = put(base / 'diagnostics/coverage_pilot_cloud_launcher/process_guard.py', {'synthetic': 'guard'})
    monkeypatch.setattr(T, 'GUARD_SHA', T.sha(guard))
    inv = put(base / 'inventory.json', dict(source_project=str(project), source_files={'sim/auscultation_pi05_v8/evaluation.py': T.sha(ef)}))
    prep = dict(runtime_project=str(project), inventory_path=str(inv), inventory_sha256=T.sha(inv))
    prep_path = put(base / 'diagnostics/horizon_pair_cloud_launcher_v1/preparation.json', prep)
    monkeypatch.setattr(T, 'PREPARATION_SHA', T.sha(prep_path))
    life, training, process, _ = lifecycle_fixture(); life['preparation'] = prep
    tr = put(run / 'training_run.json', training); life['training_receipt_sha256'] = T.sha(tr)
    life_path = put(base / 'lifecycle/lifecycle.json', life)
    put(life_path.with_name('training.process.json'), process)
    put(life_path.with_name('cpu_preflight.process.json'), life['stages'][0])
    manifest = put(base / 'dataset/manifest.json', {'synthetic': 'dataset'})
    assets = put(base / 'public_checkpoint_verified.json', {'synthetic': 'assets'})
    tokenizer = put(base / 'public-assets/paligemma_tokenizer.model', {'synthetic': 'tokenizer'})
    ai = put(base / 'runs/horizon-pair-original8-v1-A/initialization.json', {'synthetic': 'A'})
    put(run / 'initialization.json', {'synthetic': 'B'})
    config = dict(assets=dict(checkpoint_receipt_sha256=T.sha(assets), tokenizer_sha256=T.sha(tokenizer)),
                  paired_initialization=T.file_binding(ai), source_sha256={'v8/evaluation.py': T.sha(ef)})
    cp = put(run / 'configuration.json', dict(config=config, manifest=str(manifest), manifest_sha256=T.sha(manifest)))
    selection, reports = selected_fixture()
    if selected:
        c = selection['candidates'][4]
        c.update(bounds=bounds(empty=True), reasons=[], eligible=True, deploy_candidate_eligible=True)
        selection.update(selected=c, status='train_fit_candidate_requires_pair_and_fixed160')
    selection['artifact_sha256'] = {str(cp): T.sha(cp), str(tr): T.sha(tr)}
    for c in selection['candidates']:
        folder = run / 'model' / ('step-%012d' % c['local_step']); binding = {}
        for name, key in (('metadata.json', 'metadata_sha256'), ('arrays.npz', 'arrays_sha256'), ('COMPLETE.json', 'complete_sha256')):
            binding[key] = T.sha(put(folder / name, {'synthetic': name}))
        rp = put(run / 'diagnostics' / str(c['local_step']) / 'report.json', reports[c['local_step']])
        c.update(checkpoint_path=str(folder), checkpoint_binding=binding, report_path=str(rp), report_sha256=T.sha(rp))
        selection['artifact_sha256'][str(rp)] = T.sha(rp)
    sp = put(run / 'pair_selection/selection.json', selection)
    lock = dict(selection=selection, selection_path=str(sp), selection_sha256=T.sha(sp))
    lp = put(run / 'pair_selection/selection.lock.json', lock)
    put(run / 'PAIR_SELECTION_LOCKED.json', {'synthetic': 'marker'})
    calls = []
    def verify_lock(path, digest):
        calls.append('selection'); assert path == lp and T.sha(path) == digest
        return copy.deepcopy(lock)
    def dataset_artifacts(identity, output):
        calls.append('dataset'); output[identity['path']] = T.sha(identity['path'])
    fake_e = SimpleNamespace(__file__=str(ef), MARKER='PAIR_SELECTION_LOCKED.json', verify_selection_lock=verify_lock)
    fake_p = SimpleNamespace(__file__=str(pf), dataset_artifacts=dataset_artifacts)
    import sim.auscultation_pi05_v8 as package
    monkeypatch.setattr(package, 'evaluation', fake_e, raising=False)
    monkeypatch.setattr(package, 'pair_admission', fake_p, raising=False)
    kwargs = dict(b_lock=lp, b_lock_sha256=T.sha(lp), lifecycle=life_path, lifecycle_sha256=T.sha(life_path),
                  specification=specification, design_review=review)
    if selected:
        plan = dict(b_selection_lock_binding=T.file_binding(lp), fixed160_rows=rows(160))
        plan_path = put(base / 'fixed/plan.json', plan)
        receipt = dict(role='B_candidate', status='finite', numerical_failure_evidence=[], bounds=bounds(fixed=True))
        receipt_path = put(base / 'fixed/receipt.json', receipt)
        ex = put(base / 'fixed/execution.json', {'synthetic': 'external exit'})
        fixed = dict(receipt=receipt, receipt_binding=T.file_binding(receipt_path), execution_binding=T.file_binding(ex))
        def verify_plan(path):
            calls.append('plan'); assert path == plan_path
            return plan, {'synthetic': 'contexts'}, {'B_candidate': lock}
        def verify_fixed(path, p, contexts, receipt_arg):
            calls.append('fixed'); assert path == plan_path and p == plan and receipt_arg == receipt_path
            return fixed
        def closure(identity, p, contexts, locks, values):
            calls.append('closure'); assert values == [fixed]
            return {str(plan_path): T.sha(plan_path), str(receipt_path): T.sha(receipt_path), str(ex): T.sha(ex)}
        fake_p._verify_plan = verify_plan; fake_p._verify_fixed = verify_fixed; fake_p.artifact_closure = closure
        kwargs.update(fixed_plan=plan_path, fixed_plan_sha256=T.sha(plan_path),
                      fixed_receipt=receipt_path, fixed_receipt_sha256=T.sha(receipt_path))
    return kwargs, calls, manifest


@pytest.mark.parametrize('selected', [False, True])
def test_full_gate_wiring_and_receipt_reconstruction_with_explicit_scientific_stubs(tmp_path, monkeypatch, selected):
    kwargs, calls, manifest = full_wiring_fixture(tmp_path, monkeypatch, selected=selected)
    result = T.evaluate(**kwargs)
    assert result['condition_satisfied'] and not result['service_eligible'] and not result['C_training_started']
    assert result['route'] == ('locked_B_fixed160' if selected else 'negative_B_selection')
    assert ('fixed' in calls) is selected and ('closure' in calls) is selected
    assert ('selection' in calls) is (not selected)
    path = tmp_path / 'saved_condition.json'; path.write_text(json.dumps(result)); digest = T.sha(path)
    assert T.verify_receipt(path, digest) == result
    manifest.write_text('{"synthetic":"changed dataset"}')
    with pytest.raises(ValueError): T.verify_receipt(path, digest)


@pytest.mark.parametrize('marker', ['late_interrupt.json', 'training.late_interrupt.json', 'cpu_preflight.late_interrupt.json'])
def test_full_gate_late_interruption_stops_before_condition(tmp_path, monkeypatch, marker):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch)
    kwargs['lifecycle'].with_name(marker).write_text('{}')
    with pytest.raises(ValueError, match='cancellation marker'): T.evaluate(**kwargs)


def test_fixed_path_without_pinned_hash_is_rejected(tmp_path, monkeypatch):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch, selected=True)
    kwargs.pop('fixed_receipt_sha256')
    with pytest.raises(ValueError, match='expected hashes'): T.evaluate(**kwargs)


@pytest.mark.parametrize('name', ['training.process.json', 'cpu_preflight.process.json'])
def test_consumed_process_bytes_cannot_rebind_in_late_closure(tmp_path, monkeypatch, name):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch)
    original = T.verify_lifecycle
    def mutate_after_check(*args):
        original(*args)
        path = kwargs['lifecycle'].with_name(name)
        value = json.loads(path.read_bytes()); value['exit_code'] = 19
        path.write_text(json.dumps(value))
    monkeypatch.setattr(T, 'verify_lifecycle', mutate_after_check)
    with pytest.raises(ValueError, match='changed during condition'):
        T.evaluate(**kwargs)


@pytest.mark.parametrize('name', ['PAIR_SELECTION_LOCKED.json', 'initialization.json'])
def test_original_verifier_inputs_keep_their_initial_bytes(tmp_path, monkeypatch, name):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch)
    import sim.auscultation_pi05_v8 as package
    original = package.evaluation.verify_selection_lock
    def mutate_after_check(*args):
        result = original(*args)
        path = T.B_RUN / name; path.write_text(path.read_text() + '\n')
        return result
    monkeypatch.setattr(package.evaluation, 'verify_selection_lock', mutate_after_check)
    with pytest.raises(ValueError, match='changed during condition'):
        T.evaluate(**kwargs)


@pytest.mark.parametrize('target', ['plan', 'receipt'])
def test_fixed_evidence_cannot_rebind_after_initial_pin(tmp_path, monkeypatch, target):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch, selected=True)
    import sim.auscultation_pi05_v8 as package
    original = package.pair_admission._verify_plan
    def mutate_after_initial_pin(path):
        p = kwargs['fixed_plan' if target == 'plan' else 'fixed_receipt']
        # Same parsed JSON but different bytes still violates the original pin.
        p.write_text(p.read_text() + '\n')
        return original(path)
    monkeypatch.setattr(package.pair_admission, '_verify_plan', mutate_after_initial_pin)
    with pytest.raises(ValueError, match='changed|differs'): T.evaluate(**kwargs)


def test_cancellation_arriving_during_finalization_is_rechecked(tmp_path, monkeypatch):
    kwargs, _, _ = full_wiring_fixture(tmp_path, monkeypatch)
    original = T.reject_cancellation; calls = []
    def late(path):
        if calls:
            Path(path).with_name('late_interrupt.json').write_text('{}')
        calls.append(1); original(path)
    monkeypatch.setattr(T, 'reject_cancellation', late)
    with pytest.raises(ValueError, match='cancellation marker'): T.evaluate(**kwargs)
    assert len(calls) == 2
