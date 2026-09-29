"""Prospective C condition, reconstructed from completed original B evidence.

This module performs CPU verification only. A true condition neither selects a
C checkpoint nor starts training, inference, a service, or physical execution.
Pure classification helpers do not certify evidence; evaluate() is the boundary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = Path('/home/william/wam-vla-v2')
B_RUN = BASE / 'runs/horizon-pair-original8-v1-B'
SPEC_SHA = 'cfcf4573ba222656735cb5e99bc521e684d02dc0513a143b2e37bd83e2eecff1'
DESIGN_REVIEW_SHA = 'ca1b1cef8adf8af103a59f497b7312b9617a184c4f38d9eba1815fe7e4b7341d'
PREPARATION_SHA = '6899020b14d176c6db0c030ab68e202c4496940aabab417b6cf2be983d6bf42e'
GUARD_SHA = '48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
EVALUATION_SHA = '1babc07f8f53c833d39d761efb077e74ec1e9a8f4a305669b1bc7c05b848c5b9'
PAIR_SHA = '95fa3bf8444d2b90df57c8afcb647fa93ed80b1c3d9b2aa378b43a57bb2999c9'
STEPS = (0, 1217, 2434, 3650, 4867, 6084, 7300, 8517, 9734,
         10950, 12167, 13384, 14600)
SCHEMA = 'wam_rational_q4_completed_B_condition_v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path, 'Canonical absolute path required')
    need(all(not p.is_symlink() for p in (path, *path.parents)), 'Symlink path forbidden')
    need(path.is_file(), 'Regular input file required')
    return path


def sha(path):
    h = hashlib.sha256()
    with canonical(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(part)
    return h.hexdigest()


def pinned(path, expected):
    path = canonical(path)
    raw = path.read_bytes()
    need(isinstance(expected, str) and hashlib.sha256(raw).hexdigest() == expected,
         'Input hash differs before parsing: ' + str(path))
    return json.loads(raw)


def file_binding(path):
    path = canonical(path)
    return dict(path=str(path), sha256=sha(path))


def add_artifact(artifacts, path, expected=None):
    path = canonical(path); actual = sha(path); name = str(path)
    need(expected is None or expected == actual, 'Evidence artifact changed: ' + name)
    need(artifacts.get(name, actual) == actual, 'Conflicting evidence identity')
    artifacts[name] = actual
    return dict(path=name, sha256=actual)


def reject_cancellation(lifecycle_path):
    parent = canonical(lifecycle_path).parent
    paths = [parent / 'late_interrupt.json', *parent.glob('*.late_interrupt.json')]
    need(not any(p.exists() or p.is_symlink() for p in paths), 'B cancellation marker present')


def classify_bounds(bounds, rows, *, fixed160=False):
    """Summarize validated raw violations; never by itself certify a trigger."""
    count_key = 'violation_count' if fixed160 else 'violating_coordinates'
    row_key, horizon_key, joint_key, value_key = (
        ('row', 'horizon', 'joint', 'value_rad') if fixed160 else
        ('row_id', 'horizon_index', 'joint_index', 'proposed_rad'))
    count = bounds[count_key]; entries = bounds['violations']
    need(type(count) is int and count == len(entries) and count >= 0,
         'Complete violation denominator required')
    need(bounds['tolerance_rad'] == 0 and bounds['all_passed'] is (count == 0),
         'Original strict raw-bound result required')
    expected_rows = 160 if fixed160 else 120
    need(len(rows) == expected_rows and bounds['coordinate_count'] == expected_rows * 70,
         'Complete original raw coordinate denominator required')
    details = []; seen = set()
    for violation in entries:
        i, h, j = (violation[k] for k in (row_key, horizon_key, joint_key))
        need(all(type(v) is int for v in (i, h, j)) and
             0 <= i < len(rows) and 0 <= h < 10 and 0 <= j < 7,
             'Invalid violation coordinate')
        need((i, h, j) not in seen, 'Duplicate violation coordinate'); seen.add((i, h, j))
        value = violation[value_key]
        low = violation['lower_rad'] if fixed160 else bounds['lower_rad'][j]
        high = violation['upper_rad'] if fixed160 else bounds['upper_rad'][j]
        need(all(type(v) in (int, float) and math.isfinite(v) for v in (value, low, high))
             and low < high and (value < low or value > high), 'Invalid finite violation')
        row = rows[i]
        details.append(dict(row=i, horizon=h, joint=j, value_rad=value,
                            side='lower' if value < low else 'upper',
                            phase=row['phase'], start=row['start'], episode_id=row['episode_id']))
    return dict(violation_count=count, exclusively_q4=bool(count and all(v['joint'] == 3 for v in details)),
                lower_count=sum(v['side'] == 'lower' for v in details),
                upper_count=sum(v['side'] == 'upper' for v in details), violations=details)


def classify_selection(selection, reports):
    need(selection['mode'] == 'B_horizon_free_approach' and selection['full_budget_complete'] is True,
         'Completed original B required')
    candidates = selection['candidates']
    need(tuple(c['local_step'] for c in candidates) == STEPS and
         selection['candidate_count'] == 13 and tuple(selection['candidate_steps']) == STEPS,
         'Full thirteen-candidate denominator required')
    need(set(reports) == set(STEPS), 'All original reports required')
    table = []; witnesses = []
    for candidate in candidates:
        step = candidate['local_step']; report = reports[step]
        need(type(step) is int and candidate['optimizer_step'] == step,
             'Original candidate clocks required')
        bounds = classify_bounds(candidate['bounds'], report['rows'])
        passed = candidate['moving_buckets']['passed']
        need(type(passed) is int and 0 <= passed <= 16, 'Original moving denominator required')
        eligible = candidate['bounds']['all_passed'] and passed == 16
        need(candidate['eligible'] is eligible and
             candidate['deploy_candidate_eligible'] is (eligible and step > 0),
             'Candidate eligibility inconsistent with original gates')
        witness = bool(step > 0 and passed == 16 and bounds['exclusively_q4'] and
                       candidate['reasons'] == ['raw_absolute_urdf_violation'])
        if witness:
            witnesses.append(step)
        table.append(dict(local_step=step, moving_passed=passed, eligible=eligible,
                          q4_only_otherwise_eligible=witness, **bounds))
    selected = selection['selected']
    if selected is None:
        need(selection['status'] == 'rejected_no_eligible_candidate' and
             not any(c['deploy_candidate_eligible'] for c in candidates), 'Negative selection inconsistent')
    else:
        need(selection['status'] == 'train_fit_candidate_requires_pair_and_fixed160' and
             selected in candidates and selected['deploy_candidate_eligible'], 'Selected B inconsistent')
    return dict(selected_step=None if selected is None else selected['local_step'],
                candidate_table=table, q4_only_motion_capable_steps=witnesses,
                negative_selection_condition=selected is None and bool(witnesses))


def verify_lifecycle(value, training, process, preparation):
    need(value.get('schema') == 'wam_horizon_pair_arm_lifecycle_v1' and
         value.get('arm') == 'B' and value.get('mode') == 'B_horizon_free_approach',
         'Original B actual lifecycle required')
    need(all(value.get(k) is True for k in ('execute', 'completed', 'training_complete', 'gpu_released'))
         and value.get('errors') == [], 'Incomplete, failed or unreleased B lifecycle')
    need(value.get('output') == str(B_RUN) and value.get('training_receipt') == training and
         value.get('preparation') == preparation and value.get('preparation_sha256') == PREPARATION_SHA,
         'Lifecycle inputs or actual training receipt differ')
    need(training.get('training_complete') is True and training.get('local_step') == 14600 and
         training.get('optimizer_step') == 14600 and training.get('schedule_cursor') == 58400 and
         training.get('interrupted') is False and training.get('error') is None,
         'Full uninterrupted B receipt required')
    stages = value.get('stages', [])
    need(len(stages) == 2 and [v.get('stage') for v in stages] == ['cpu_preflight', 'training']
         and stages[1] == process, 'Original full owned training stage required')
    for stage in stages:
        need(stage.get('exit_code') == 0 and type(stage.get('exit_code')) is int and
             stage.get('passed') is True and stage.get('process_group_drained') is True and
             all(stage.get(k) == [] for k in ('errors', 'signals', 'deferred_signals')),
             'Actual clean child exit and drain required')
        identity = stage.get('process_identity', {})
        pid = identity.get('pid')
        need(type(pid) is int and pid > 0 and identity.get('pgid') == pid and
             identity.get('sid') == pid and str(identity.get('start_ticks', '')).isdigit(),
             'Actual owned child identity required')
    need(process['process_identity']['pid'] == training['pid'] and
         process['command'] == value['command'], 'Actual trainer identity differs')
    expected = [str(BASE / 'vendor/openpi/.venv/bin/python'), '-u', '-B', '-m',
                'sim.auscultation_pi05_v8.train', '--manifest',
                str(BASE / 'datasets/pi05_v6_task_pose_armature02_v1/task_conditioned_bound/manifest.json'),
                '--mode', 'B_horizon_free_approach', '--output', str(B_RUN),
                '--paired-initialization', str(BASE / 'runs/horizon-pair-original8-v1-A/initialization.json')]
    resume = training.get('resume')
    if resume is not None:
        path = Path(resume)
        need(path.is_absolute() and path.resolve() == path and
             path.parent in (B_RUN / 'model', B_RUN / 'recovery') and
             path.name.startswith('step-') and path.name[5:].isdigit() and len(path.name[5:]) == 12,
             'Only explicitly recorded same-run B resume may precede completion')
        canonical(path / 'COMPLETE.json')
        expected += ['--resume', str(path)]
    need(process['command'] == expected and stages[0]['command'] == expected + ['--verify-only'],
         'Exact original B training and CPU preflight argv required')


def evaluate(*, b_lock, b_lock_sha256, lifecycle, lifecycle_sha256,
             specification, design_review, fixed_plan=None, fixed_receipt=None,
             fixed_plan_sha256=None, fixed_receipt_sha256=None):
    """Fresh original B reconstruction; this function never starts any work."""
    spec = pinned(specification, SPEC_SHA); review = pinned(design_review, DESIGN_REVIEW_SHA)
    need(spec.get('schema') == 'wam_q4_rational_conditional_experiment_design_v1' and
         spec.get('current_AB_immutable') is True, 'Exact prospective design required')
    need(review.get('passed') is True, 'Independent design review required')
    b_lock = canonical(b_lock)
    need(b_lock == B_RUN / 'pair_selection/selection.lock.json', 'Original B selection required')
    saved_lock = pinned(b_lock, b_lock_sha256)
    fixed_inputs = (fixed_plan, fixed_receipt, fixed_plan_sha256, fixed_receipt_sha256)
    need(all(v is None for v in fixed_inputs) or all(v is not None for v in fixed_inputs),
         'Both actual fixed160 paths and expected hashes required')
    from sim.auscultation_pi05_v8 import evaluation as E
    from sim.auscultation_pi05_v8 import pair_admission as P
    need(sha(Path(E.__file__).resolve()) == EVALUATION_SHA, 'Original B evaluation source differs')
    need(sha(Path(P.__file__).resolve()) == PAIR_SHA, 'Original pair verifier differs')
    # Capture unpinned original inputs before the scientific verifier consumes
    # them. Later closure construction must never adopt replacement bytes.
    artifacts = {}
    for path, expected in ((B_RUN / E.MARKER, None), (B_RUN / 'initialization.json', None),
                           (Path(__file__).resolve(), None),
                           (Path(E.__file__).resolve(), EVALUATION_SHA),
                           (Path(P.__file__).resolve(), PAIR_SHA)):
        add_artifact(artifacts, path, expected)
    fixed = None; plan = None; expected_plan_binding = expected_receipt_binding = None
    if fixed_plan is None:
        lock = E.verify_selection_lock(b_lock, b_lock_sha256)
    else:
        original_plan = pinned(fixed_plan, fixed_plan_sha256)
        original_receipt = pinned(fixed_receipt, fixed_receipt_sha256)
        expected_plan_binding = dict(path=str(canonical(fixed_plan)), sha256=fixed_plan_sha256)
        expected_receipt_binding = dict(path=str(canonical(fixed_receipt)), sha256=fixed_receipt_sha256)
        plan, contexts, locks = P._verify_plan(canonical(fixed_plan))
        need(plan == original_plan, 'Fixed160 plan changed after initial pin')
        need(plan['b_selection_lock_binding'] == dict(path=str(b_lock), sha256=b_lock_sha256),
             'Fixed160 plan selected another B')
        lock = locks['B_candidate']
        fixed = P._verify_fixed(canonical(fixed_plan), plan, contexts, canonical(fixed_receipt))
        need(fixed['receipt_binding'] == expected_receipt_binding and fixed['receipt'] == original_receipt,
             'Fixed160 receipt changed after initial pin')
    need(lock == saved_lock, 'Fresh B lock differs')
    selection = lock['selection']
    for path, expected in selection['artifact_sha256'].items():
        add_artifact(artifacts, path, expected)
    reports = {}
    for candidate in selection['candidates']:
        path = candidate['report_path']
        reports[candidate['local_step']] = pinned(path, candidate['report_sha256'])
        cb = candidate['checkpoint_binding']; folder = Path(candidate['checkpoint_path'])
        for name, key in (('metadata.json', 'metadata_sha256'), ('arrays.npz', 'arrays_sha256'),
                          ('COMPLETE.json', 'complete_sha256')):
            add_artifact(artifacts, folder / name, cb[key])
    # Retain complete original B qualification/source ancestry, not only reports.
    configuration_path = B_RUN / 'configuration.json'
    configuration = pinned(configuration_path, selection['artifact_sha256'][str(configuration_path)])
    config = configuration['config']
    P.dataset_artifacts(dict(path=configuration['manifest'], sha256=configuration['manifest_sha256']), artifacts)
    add_artifact(artifacts, BASE / 'public_checkpoint_verified.json', config['assets']['checkpoint_receipt_sha256'])
    add_artifact(artifacts, BASE / 'public-assets/paligemma_tokenizer.model', config['assets']['tokenizer_sha256'])
    add_artifact(artifacts, config['paired_initialization']['path'], config['paired_initialization']['sha256'])
    add_artifact(artifacts, lock['selection_path'], lock['selection_sha256'])
    source_root = Path(E.__file__).resolve().parents[2]
    for name, expected in config['source_sha256'].items():
        version, rest = name.split('/', 1)
        need(version in ('v6', 'v7', 'v8') and '/' not in rest and rest.endswith('.py'),
             'Unexpected B training source key')
        add_artifact(artifacts, source_root / ('sim/auscultation_pi05_' + version) / rest, expected)
    result = classify_selection(selection, reports)
    life = pinned(lifecycle, lifecycle_sha256); life_path = canonical(lifecycle)
    reject_cancellation(life_path)
    process_path = life_path.with_name('training.process.json')
    cpu_path = life_path.with_name('cpu_preflight.process.json')
    process_binding = add_artifact(artifacts, process_path)
    cpu_binding = add_artifact(artifacts, cpu_path)
    process = pinned(process_path, process_binding['sha256'])
    cpu_process = pinned(cpu_path, cpu_binding['sha256'])
    training_path = B_RUN / 'training_run.json'; training = pinned(training_path, life['training_receipt_sha256'])
    add_artifact(artifacts, training_path, life['training_receipt_sha256'])
    preparation_path = BASE / 'diagnostics/horizon_pair_cloud_launcher_v1/preparation.json'
    preparation = pinned(preparation_path, PREPARATION_SHA)
    verify_lifecycle(life, training, process, preparation)
    need(cpu_process == life['stages'][0],
         'Actual CPU preflight record differs')
    inventory = pinned(preparation['inventory_path'], preparation['inventory_sha256'])
    need(inventory['source_project'] == preparation['runtime_project'], 'B training snapshot differs')
    for name, expected in inventory['source_files'].items():
        add_artifact(artifacts, Path(inventory['source_project']) / name, expected)
    add_artifact(artifacts, preparation['inventory_path'], preparation['inventory_sha256'])
    for path, expected in ((b_lock, b_lock_sha256), (life_path, lifecycle_sha256),
                           (specification, SPEC_SHA), (design_review, DESIGN_REVIEW_SHA),
                           (preparation_path, PREPARATION_SHA),
                           (BASE / 'diagnostics/coverage_pilot_cloud_launcher/process_guard.py', GUARD_SHA)):
        add_artifact(artifacts, path, expected)
    trigger = result['negative_selection_condition']; route = 'negative_B_selection' if trigger else 'not_triggered'
    fixed_summary = None
    if fixed is not None:
        receipt = fixed['receipt']
        need(result['selected_step'] is not None and receipt['role'] == 'B_candidate',
             'Original locked B candidate fixed160 required')
        need(receipt.get('status') == 'finite' and receipt.get('numerical_failure_evidence') == []
             and receipt.get('bounds') is not None,
             'Nonfinite or invalid fixed160 output is not this condition')
        fixed_summary = classify_bounds(receipt['bounds'], plan['fixed160_rows'], fixed160=True)
        trigger = fixed_summary['exclusively_q4']; route = 'locked_B_fixed160' if trigger else 'not_triggered'
        # Reuse the original full dependency closure, including external exit evidence.
        closure = P.artifact_closure(expected_plan_binding, plan, contexts, locks, [fixed])
        for path, expected in closure.items():
            add_artifact(artifacts, path, expected)
        add_artifact(artifacts, Path(P.__file__).resolve(), PAIR_SHA)
        add_artifact(artifacts, fixed_plan, fixed_plan_sha256)
        add_artifact(artifacts, fixed_receipt, fixed_receipt_sha256)
    for path, expected in artifacts.items():
        need(sha(path) == expected, 'Evidence changed during condition reconstruction')
    reject_cancellation(life_path)
    return dict(schema=SCHEMA,
                specification_binding=dict(path=str(canonical(specification)), sha256=SPEC_SHA),
                b_selection_lock_binding=dict(path=str(b_lock), sha256=b_lock_sha256),
                lifecycle_binding=dict(path=str(life_path), sha256=lifecycle_sha256),
                design_review_binding=dict(path=str(canonical(design_review)), sha256=DESIGN_REVIEW_SHA),
                fixed_plan_binding=expected_plan_binding,
                fixed_receipt_binding=expected_receipt_binding,
                condition_satisfied=bool(trigger), route=route, selection=result, fixed160=fixed_summary,
                artifact_sha256=artifacts, software_preparation_only=True, C_training_started=False,
                C_selected=False, service_eligible=False, physical_accepted=False)


def verify_receipt(path, expected_sha256):
    saved = pinned(path, expected_sha256)
    need(saved.get('schema') == SCHEMA, 'Original completed-B condition receipt required')
    kwargs = dict(b_lock=saved['b_selection_lock_binding']['path'],
                  b_lock_sha256=saved['b_selection_lock_binding']['sha256'],
                  lifecycle=saved['lifecycle_binding']['path'],
                  lifecycle_sha256=saved['lifecycle_binding']['sha256'],
                  specification=saved['specification_binding']['path'],
                  design_review=saved['design_review_binding']['path'])
    if saved['fixed_plan_binding'] is not None:
        kwargs.update(fixed_plan=saved['fixed_plan_binding']['path'],
                      fixed_plan_sha256=saved['fixed_plan_binding']['sha256'],
                      fixed_receipt=saved['fixed_receipt_binding']['path'],
                      fixed_receipt_sha256=saved['fixed_receipt_binding']['sha256'])
    rebuilt = evaluate(**kwargs)
    need(saved == rebuilt and rebuilt['condition_satisfied'] is True and sha(path) == expected_sha256,
         'Fresh completed-B condition differs or is not satisfied')
    return rebuilt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('b-lock', 'lifecycle', 'specification', 'design-review'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--b-lock-sha256', required=True); p.add_argument('--lifecycle-sha256', required=True)
    p.add_argument('--fixed-plan', type=Path); p.add_argument('--fixed-receipt', type=Path)
    p.add_argument('--fixed-plan-sha256'); p.add_argument('--fixed-receipt-sha256')
    args = vars(p.parse_args()); result = evaluate(**args)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    if not result['condition_satisfied']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
