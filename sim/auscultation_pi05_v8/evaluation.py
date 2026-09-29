"""Paired codec TRAIN-fit reports and immutable candidate selection.

The numerical rows, raw bounds, movement gates and ranking are the unchanged
V7 algorithms. New metadata identifies fresh initialization and the exact codec
arm. Selection alone is neither physical acceptance nor service admission.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as parent
from sim.auscultation_pi05_v7.coverage_pilot_schedule import digest, file_sha, json_bytes
from sim.auscultation_pi05_v7.coverage_pilot_selection import pick_candidate, RULE, EXPECTED_STEPS
from . import action_encoding as codec

SCHEMA='wam_horizon_pair_train_numeric_report_v1'
BINDING_SCHEMA='wam_horizon_pair_checkpoint_binding_v1'
BINDING_KEYS=frozenset(('schema','mode','checkpoint_path','local_step','optimizer_step',
    'metadata_sha256','arrays_sha256','complete_sha256','data_hash','normalization_sha256',
    'config_sha256','schedule_sha256','source_sha256'))
SELECTION_SCHEMA='wam_horizon_pair_train_fit_selection_v1'
LOCK_SCHEMA='wam_horizon_pair_train_fit_selection_lock_v1'
MARKER='PAIR_SELECTION_LOCKED.json'
SAMPLER=parent.SAMPLER
metrics=parent.metrics
need=parent.need
read=parent.read
write_new=parent.write_new


def build_diagnostic_schedule(dataset):
    codec.action_encoding_binding(dataset.mode)
    return parent.build_diagnostic_schedule(dataset.parent)


def references(dataset,rows):
    return parent.references(dataset.parent,rows)


def verify_checkpoint_binding(binding):
    need(isinstance(binding,dict) and set(binding)==BINDING_KEYS,'Exact paired checkpoint fields required')
    need(binding['schema']==BINDING_SCHEMA,'Paired checkpoint schema required')
    codec.action_encoding_binding(binding['mode'])
    need(type(binding['local_step']) is int and binding['local_step']>=0 and
         type(binding['optimizer_step']) is int and binding['optimizer_step']==binding['local_step'],
         'Fresh-base optimizer and local clocks must agree')
    from .train import checkpoint_binding
    need(checkpoint_binding(binding['checkpoint_path'])==binding,'Checkpoint bytes/source binding changed')


def _numeric(dataset,predicted,normalized,binding):
    verify_checkpoint_binding(binding)
    need(binding['mode']==dataset.mode and binding['data_hash']==dataset.data_hash and
         binding['normalization_sha256']==digest(dataset.normalization),'Dataset/mode/normalization binding changed')
    rows=build_diagnostic_schedule(dataset);ref,anchor=references(dataset,rows)
    p,z=map(np.asarray,(predicted,normalized))
    need(p.shape==z.shape==(120,10,7) and p.dtype.kind==z.dtype.kind=='f' and
         np.isfinite(p).all() and np.isfinite(z).all(),'Finite 120x10x7 predictions required')
    decoded=np.stack([codec.decode_chunk(value,q,row['phase'],dataset.normalization['residual_actions'],mode=dataset.mode)
                      for value,q,row in zip(z,anchor,rows)])
    need(p.dtype==decoded.dtype and p.tobytes()==decoded.tobytes(),'Raw proposals differ from exact paired codec decode')
    return rows,ref,anchor,metrics(p,ref,anchor,rows)


def _report(binding,rows,result,numeric_sha):
    return dict(schema=SCHEMA,mode=binding['mode'],checkpoint_binding=binding,sampler=SAMPLER,
        rows=rows,diagnostic_schedule_sha256=digest(rows),numeric_sha256=numeric_sha,metrics=result,
        local_step=binding['local_step'],optimizer_step=binding['optimizer_step'],
        initialization_baseline_only=binding['local_step']==0,train_predictions_only=True,
        heldout_predictions_generated=False,closed_loop_validated=False)


def _report_path(binding):
    return Path(binding['checkpoint_path']).parents[1]/'diagnostics'/('local_%06d'%binding['local_step'])/'report.json'


def save_checkpoint_report(output,dataset,predictions,normalized_predictions,checkpoint_binding):
    rows,ref,anchor,result=_numeric(dataset,predictions,normalized_predictions,checkpoint_binding)
    output=Path(output).resolve()
    need(output==_report_path(checkpoint_binding).parent.resolve(),'Original candidate report directory required')
    output.mkdir(parents=True,exist_ok=False);numeric=output/'predictions.npz'
    with numeric.open('xb') as f:
        np.savez_compressed(f,predictions=predictions,normalized_predictions=normalized_predictions,reference=ref,anchors=anchor)
        f.flush();os.fsync(f.fileno())
    verify_checkpoint_binding(checkpoint_binding)
    report=_report(checkpoint_binding,rows,result,file_sha(numeric));write_new(output/'report.json',report)
    return report


def verify_checkpoint_report(path,dataset,checkpoint_binding):
    path=Path(path);path=path/'report.json' if path.is_dir() else path
    need(path.resolve()==_report_path(checkpoint_binding).resolve(),'Original candidate report path required')
    saved=read(path);numeric=path.with_name('predictions.npz');before=file_sha(numeric)
    need(saved.get('numeric_sha256')==before,'Numeric SHA differs')
    with np.load(numeric,allow_pickle=False) as f:
        need(set(f.files)=={'predictions','normalized_predictions','reference','anchors'},'Numeric schema differs')
        rows,ref,anchor,result=_numeric(dataset,f['predictions'],f['normalized_predictions'],checkpoint_binding)
        need(np.array_equal(f['reference'],ref) and np.array_equal(f['anchors'],anchor),'Original labels/anchors changed')
    expected=_report(checkpoint_binding,rows,result,before)
    need(saved==expected and file_sha(numeric)==before,'Report or numeric evidence differs from reconstruction')
    return expected


def choose_result(candidates,mode):
    codec.action_encoding_binding(mode)
    need(all(type(c['local_step']) is int for c in candidates),'Integer candidate clocks required')
    need(tuple(c['local_step'] for c in candidates)==EXPECTED_STEPS,'Full ordered thirteen-candidate denominator required')
    # Validate every score/eligibility even when no candidate is feasible.
    for c in candidates:
        need(type(c['eligible']) is bool,'Boolean eligibility required')
        need(all(type(c['scores'][k]) in (int,float) and np.isfinite(c['scores'][k]) and c['scores'][k]>=0
                 for k in ('worst_initial_first_max_abs_rad','macro_first_mae_rad','macro_full_mae_rad')),
             'Finite nonnegative ranking scores required')
    selected=pick_candidate(candidates) if any(c['eligible'] and c['local_step']>0 for c in candidates) else None
    return dict(selected=selected,status='train_fit_candidate_requires_pair_and_fixed160' if selected else 'rejected_no_eligible_candidate',
                descriptive_only_local_step=14600 if selected is None and mode==codec.A else None,
                service_eligible=False)


def load_context(run):
    from .train import verified_run_context
    return verified_run_context(Path(run).resolve(strict=True))


def build_selection(context):
    from .train import checkpoint_binding
    config=context['config'];run=Path(config['training_run_path']).resolve(strict=True);data=context['dataset']
    need(context.get('completed') is True and config['mode']==data.mode,'Complete matching-arm training required')
    need(tuple(context['schedule']['candidate_steps'])==EXPECTED_STEPS and context['schedule']['total_steps']==14600,
         'Complete declared 14600 steps required')
    paths=[Path(p).resolve(strict=True) for p in context['candidate_paths']]
    need(paths==[run/'model'/('step-%012d'%s) for s in EXPECTED_STEPS],'Original thirteen candidate paths required')
    artifacts={str(run/name):file_sha(run/name) for name in
               ('configuration.json','training_run.json','diagnostic_schedule.json','training_schedule.json')}
    rows=build_diagnostic_schedule(data)
    need(read(run/'diagnostic_schedule.json')==dict(data_hash=data.data_hash,rows=rows),'Predeclared diagnostic origins changed')
    candidates=[]
    for step,path in zip(EXPECTED_STEPS,paths):
        binding=checkpoint_binding(path)
        need(binding['local_step']==binding['optimizer_step']==step and binding['mode']==data.mode and
             binding['config_sha256']==digest(config) and binding['source_sha256']==config['source_sha256'],
             'Candidate clock, arm or source differs')
        report_path=_report_path(binding);report=verify_checkpoint_report(report_path,data,binding);m=report['metrics']
        for p in (report_path,report_path.with_name('predictions.npz')):artifacts[str(p)]=file_sha(p)
        reasons=[]
        if not m['bounds']['all_passed']:reasons.append('raw_absolute_urdf_violation')
        if m['moving_buckets']['passed']!=16:reasons.append('moving_bucket_failed_or_unassessable')
        if step==0:reasons.append('fresh_initialization_baseline_not_candidate')
        candidates.append(dict(local_step=step,optimizer_step=step,checkpoint_path=str(path),checkpoint_binding=binding,
            eligible=m['eligible'],deploy_candidate_eligible=m['eligible'] and step>0,reasons=reasons,
            scores=m['scores'],bounds=m['bounds'],moving_buckets=m['moving_buckets'],
            report_path=str(report_path),report_sha256=artifacts[str(report_path)]))
    result=choose_result(candidates,data.mode)
    for p,h in artifacts.items():need(file_sha(p)==h,'Evidence changed during selection')
    for c in candidates:need(checkpoint_binding(c['checkpoint_path'])==c['checkpoint_binding'],'Checkpoint changed during selection')
    return dict(schema=SELECTION_SCHEMA,run=str(run),mode=data.mode,criterion=RULE,
        source_sha256=config['source_sha256'],candidate_steps=list(EXPECTED_STEPS),candidate_count=13,
        full_budget_complete=True,artifact_sha256=artifacts,diagnostic_schedule_sha256=digest(rows),candidates=candidates,
        **result,heldout_predictions_generated=False,generalization_validated=False,closed_loop_validated=False,
        limitations='Original eight TRAIN only. Both paired budgets, fixed160 diagnostic and separately verified physical execution remain required.')


def _marker(path):
    return dict(schema='wam_horizon_pair_selection_marker_v1',selection_lock=str(path),selection_lock_sha256=file_sha(path))


def _lock(selection,path):
    return dict(schema=LOCK_SCHEMA,selection_path=str(path),selection_sha256=file_sha(path),selection=selection,
                scope='Frozen TRAIN-fit outcome, including negative outcome. Not service or physical admission.')


def verify_selection_lock(path,expected_sha=None):
    path=Path(path).resolve(strict=True);before=file_sha(path)
    need(expected_sha is None or before==expected_sha,'Lock SHA differs')
    lock=read(path);need(lock.get('schema')==LOCK_SCHEMA,'Paired selection lock required')
    run=Path(lock['selection']['run']).resolve(strict=True)
    need(path==run/'pair_selection/selection.lock.json' and read(run/MARKER)==_marker(path),'Original run/marker required')
    selection_path=path.with_name('selection.json')
    need(lock['selection_path']==str(selection_path) and lock['selection_sha256']==file_sha(selection_path),'Selection SHA differs')
    rebuilt=build_selection(load_context(run))
    need(rebuilt==read(selection_path)==lock['selection'] and _lock(rebuilt,selection_path)==lock and file_sha(path)==before,
         'Fresh paired selection reconstruction differs')
    return lock


def select_run(run):
    run=Path(run).resolve(strict=True);context=load_context(run)
    need(context['config']['training_run_path']==str(run),'Copied run cannot select')
    need(not any((run/n).exists() for n in (MARKER,'SELECTION_LOCKED.json','FINAL_TEST_STARTED.json')),'Run already consumed')
    output=run/'pair_selection';need(not output.exists(),'Fresh selection output required')
    write_new(run/MARKER,dict(schema='wam_horizon_pair_selection_reserved_v1',run=str(run)))
    selection=build_selection(context);output.mkdir();path=output/'selection.json';write_new(path,selection)
    lock_path=output/'selection.lock.json';write_new(lock_path,_lock(selection,path));lock_path.chmod(0o444)
    pending=run/(MARKER+'.completed.tmp');write_new(pending,_marker(lock_path));os.replace(pending,run/MARKER)
    return dict(selection_lock=str(lock_path),selection_lock_sha256=file_sha(lock_path),status=selection['status'],
        selected_local_step=None if selection['selected'] is None else selection['selected']['local_step'],
        descriptive_only_local_step=selection['descriptive_only_local_step'],service_eligible=False,closed_loop_validated=False)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(select_run(args.run),allow_nan=False),flush=True)


if __name__=='__main__':main()
