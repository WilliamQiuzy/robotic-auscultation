"""Lock one TRAIN-fit pilot candidate after the full declared coverage budget.

This is readiness for a separate actual R4 trial, not heldout selection or a
generalization/physical success claim. Step zero is a diagnostic baseline only.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from . import coverage_pilot_evaluation as E
from .coverage_pilot_schedule import source_hashes

SCHEMA='wam_coverage_pilot_train_fit_selection_v1'
LOCK_SCHEMA='wam_coverage_pilot_train_fit_selection_lock_v1'
MARKER='PILOT_SELECTION_LOCKED.json'
EXPECTED_STEPS=(0,1217,2434,3650,4867,6084,7300,8517,9734,10950,12167,13384,14600)
RULE='all raw URDF + all16 moving FREE/APPROACH buckets; worst8 initial first maxabs, macro40 first MAE, macro40 full MAE, earlier positive local step'


def pick_candidate(candidates):
    feasible=[];steps=set()
    for candidate in candidates:
        step=candidate['local_step'];E.need(type(step) is int and step>=0 and step not in steps,'Unique nonnegative local steps required');steps.add(step)
        E.need(type(candidate['eligible']) is bool,'Boolean eligibility required')
        scores=candidate['scores']
        E.need(all(type(scores[k]) in (int,float) and np.isfinite(scores[k]) and scores[k]>=0 for k in
            ('worst_initial_first_max_abs_rad','macro_first_mae_rad','macro_full_mae_rad')),'Finite nonnegative ranking scores required')
        if candidate['eligible'] and step>0:feasible.append(candidate)
    E.need(feasible,'No eligible positive-step pilot candidate')
    return min(feasible,key=lambda c:(c['scores']['worst_initial_first_max_abs_rad'],c['scores']['macro_first_mae_rad'],
        c['scores']['macro_full_mae_rad'],c['local_step']))


def load_context(run):
    from .coverage_pilot_train import verified_run_context
    return verified_run_context(Path(run).resolve(strict=True))


def build_selection(context):
    from .coverage_pilot_train import checkpoint_binding
    config=context['config'];run=Path(config['training_run_path']).resolve(strict=True)
    E.need(context.get('completed') is True,'Actual complete pilot budget required')
    E.need(tuple(context['schedule']['candidate_steps'])==EXPECTED_STEPS and context['schedule']['total_steps']==14600,
           'Full declared 14600-step / thirteen-candidate denominator required')
    E.need(context['pilot_source_sha256']==source_hashes(),'Pilot source changed')
    paths=[Path(p).resolve(strict=True) for p in context['candidate_paths']]
    E.need(paths==[run/'model'/('step-%012d'%s) for s in EXPECTED_STEPS],'Exact original-run candidate namespace required')
    artifacts={str(run/'configuration.json'):E.file_sha(run/'configuration.json'),
        str(run/'training_run.json'):E.file_sha(run/'training_run.json'),
        str(run/'diagnostic_schedule.json'):E.file_sha(run/'diagnostic_schedule.json')}
    rows=E.build_diagnostic_schedule(context['dataset'])
    E.need(E.read(run/'diagnostic_schedule.json')==dict(data_hash=context['dataset'].data_hash,rows=rows),'Pretraining diagnostic schedule changed')
    candidates=[]
    for step,path in zip(EXPECTED_STEPS,paths):
        binding=checkpoint_binding(path)
        E.need(binding['local_step']==step and binding['optimizer_step']==4000+step and
            binding['config_sha256']==E.digest(config) and binding['pilot_binding']==context['pilot_binding'] and
            binding['pilot_source_sha256']==context['pilot_source_sha256'],'Candidate configuration/ancestry differs')
        report_path=run/'diagnostics'/('local_%06d'%step)/'report.json'
        report=E.verify_checkpoint_report(report_path,context['dataset'],binding);metrics=report['metrics']
        for p in (report_path,report_path.with_name('predictions.npz')):artifacts[str(p)]=E.file_sha(p)
        reasons=[]
        if not metrics['bounds']['all_passed']:reasons.append('raw_absolute_urdf_violation')
        if metrics['moving_buckets']['passed']!=16:reasons.append('moving_bucket_failed_or_unassessable')
        if step==0:reasons.append('initialization_baseline_not_repaired_candidate')
        candidates.append(dict(local_step=step,optimizer_step=4000+step,checkpoint_path=str(path),checkpoint_binding=binding,
            eligible=metrics['eligible'],deploy_candidate_eligible=metrics['eligible'] and step>0,
            reasons=reasons,scores=metrics['scores'],bounds=metrics['bounds'],moving_buckets=metrics['moving_buckets'],
            report_path=str(report_path),report_sha256=artifacts[str(report_path)]))
    try:selected=pick_candidate(candidates)
    except ValueError:selected=None
    E.need(source_hashes()==context['pilot_source_sha256'],'Pilot source changed during selection')
    for path,digest in artifacts.items():E.need(E.file_sha(path)==digest,'Diagnostic evidence changed during selection')
    for candidate in candidates:E.need(checkpoint_binding(candidate['checkpoint_path'])==candidate['checkpoint_binding'],'Checkpoint changed during selection')
    return dict(schema=SCHEMA,run=str(run),criterion=RULE,pilot_binding=context['pilot_binding'],
        pilot_source_sha256=context['pilot_source_sha256'],candidate_steps=list(EXPECTED_STEPS),candidate_count=13,
        full_budget_complete=True,artifact_sha256=artifacts,diagnostic_schedule_sha256=E.digest(rows),
        candidates=candidates,selected=selected,status='train_fit_ready_for_separate_R4_trial' if selected else 'rejected_no_eligible_candidate',
        heldout_predictions_generated=False,generalization_validated=False,closed_loop_validated=False,
        limitations='Original eight TRAIN episodes only. Actual observed progress/contact/HOLD/release and all physical gates remain required.')


def _marker(path):
    return dict(schema='wam_coverage_pilot_selection_marker_v1',selection_lock=str(path),selection_lock_sha256=E.file_sha(path))


def _lock(selection,selection_path):
    E.need(selection['selected'] is not None,'No eligible train-fit candidate to lock')
    return dict(schema=LOCK_SCHEMA,selection_path=str(selection_path),selection_sha256=E.file_sha(selection_path),
        selection=selection,scope='train-fit pilot R4 readiness only; no fresh generalization or closed-loop acceptance')


def verify_selection_lock(path,expected_sha=None):
    path=Path(path).resolve(strict=True);before=E.file_sha(path)
    if expected_sha is not None:E.need(before==expected_sha,'Selection lock SHA differs')
    lock=E.read(path);E.need(lock.get('schema')==LOCK_SCHEMA,'Pilot train-fit lock schema required')
    run=Path(lock['selection']['run']).resolve(strict=True)
    E.need(path==run/'pilot_selection'/'selection.lock.json','Original run-scoped lock required')
    E.need(E.read(run/MARKER)==_marker(path),'Original-run selection marker differs')
    selection_path=path.with_name('selection.json')
    E.need(lock['selection_path']==str(selection_path) and lock['selection_sha256']==E.file_sha(selection_path),'Selection evidence binding differs')
    rebuilt=build_selection(load_context(run))
    E.need(rebuilt==E.read(selection_path)==lock['selection'] and _lock(rebuilt,selection_path)==lock,'Fresh train-fit selection differs')
    E.need(E.file_sha(path)==before,'Selection lock changed during verification')
    return lock


def select_run(run):
    run=Path(run).resolve(strict=True);context=load_context(run)
    E.need(context['config']['training_run_path']==str(run),'Copied run cannot create another lock')
    E.need(not any((run/n).exists() for n in (MARKER,'SELECTION_LOCKED.json','FINAL_TEST_STARTED.json')),'Run already locked or consumed')
    output=run/'pilot_selection';E.need(not output.exists(),'Fresh pilot selection output required')
    # Exclusive fail-closed reservation also blocks same-run training resume.
    E.write_new(run/MARKER,dict(schema='wam_coverage_pilot_selection_reserved_v1',run=str(run)))
    result=build_selection(context);output.mkdir(exist_ok=False)
    selection_path=output/'selection.json';E.write_new(selection_path,result)
    E.need(result['selected'] is not None,'No eligible candidate; rejection evidence and reservation retained')
    path=output/'selection.lock.json';E.write_new(path,_lock(result,selection_path));path.chmod(0o444)
    pending=run/(MARKER+'.completed.tmp');E.write_new(pending,_marker(path));os.replace(pending,run/MARKER)
    return dict(selection_lock=str(path),selection_lock_sha256=E.file_sha(path),selected_local_step=result['selected']['local_step'],
        status=result['status'],closed_loop_validated=False,generalization_validated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--verify-lock',type=Path);args=parser.parse_args()
    if args.verify_lock:
        lock=verify_selection_lock(args.verify_lock);E.need(Path(lock['selection']['run'])==args.run.resolve(),'Run differs from lock')
        result=dict(verified=True,selected_local_step=lock['selection']['selected']['local_step'],closed_loop_validated=False)
    else:result=select_run(args.run)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
