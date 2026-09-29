"""Conditional rational-q4 TRAIN-fit reports and immutable selection.

The numerical rows, raw bounds, movement gates and ranking are the unchanged
V7 algorithms. New metadata identifies fresh initialization and the exact codec
arm. Selection alone is neither physical acceptance nor service admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sim.auscultation_pi05_v8 import evaluation as parent
from sim.auscultation_pi05_v7.coverage_pilot_schedule import digest, file_sha, json_bytes
from sim.auscultation_pi05_v7.coverage_pilot_selection import pick_candidate, RULE, EXPECTED_STEPS
from . import action_encoding as codec

SCHEMA='wam_rational_q4_train_numeric_report_v1'
BINDING_SCHEMA='wam_rational_q4_checkpoint_binding_v1'
BINDING_KEYS=frozenset(('schema','mode','checkpoint_path','local_step','optimizer_step',
    'metadata_sha256','arrays_sha256','complete_sha256','data_hash','normalization_sha256',
    'config_sha256','schedule_sha256','source_sha256'))
SELECTION_SCHEMA='wam_rational_q4_train_fit_selection_v1'
LOCK_SCHEMA='wam_rational_q4_train_fit_selection_lock_v1'
MARKER='RATIONAL_SELECTION_LOCKED.json'
SAMPLER=parent.SAMPLER
CONTEXT_FILES=('configuration.json','training_run.json','diagnostic_schedule.json',
               'training_schedule.json','initialization.json')
metrics=parent.metrics
need=parent.need
read=parent.read
write_new=parent.write_new


def read_pinned(path,expected_sha=None):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'Regular JSON evidence required')
    raw=path.read_bytes();identity=hashlib.sha256(raw).hexdigest()
    need(expected_sha is None or expected_sha==identity,'JSON evidence hash differs before parsing')
    return json.loads(raw),identity


def build_diagnostic_schedule(dataset):
    codec.action_encoding_binding(dataset.mode)
    return parent.build_diagnostic_schedule(dataset.parent)


def references(dataset,rows):
    return parent.references(dataset.parent,rows)


def q4_slopes(residual,anchor):
    """Exact-real decoder derivative summary; float32 steps are not derivatives."""
    r,a=np.broadcast_arrays(np.asarray(residual,np.float64),np.asarray(anchor,np.float64))
    need(np.isfinite(r).all() and np.isfinite(a).all() and
         np.all((a>=0)&(a<=codec.U_SAFE_RAD)), 'Finite in-domain derivative inputs required')
    lower_zero=(r==0)&(a==0);upper_zero=(r==0)&(a==codec.U_SAFE_RAD)
    undefined=lower_zero|upper_zero;slopes=np.ones(r.shape,np.float64)
    below=r<0;above=r>0
    # Algebraically equivalent ratios avoid squaring a huge denominator.
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        slopes[below]=(a[below]/(a[below]-r[below]))**2
        b=codec.U_SAFE_RAD-a[above]
        slopes[above]=(b/(b+r[above]))**2
    finite=slopes[~undefined]
    need(np.isfinite(finite).all() and np.all((finite>=0)&(finite<=1)), 'Invalid exact-real slope')
    return dict(finite_count=int(finite.size),undefined_count=int(undefined.sum()),
        zero_at_lower_endpoint_count=int(lower_zero.sum()),zero_at_upper_endpoint_count=int(upper_zero.sum()),
        lower_endpoint_zero_left_right_limits=[0.,1.],upper_endpoint_zero_left_right_limits=[1.,0.],
        min=None if not finite.size else float(finite.min()),
        p01=None if not finite.size else float(np.quantile(finite,.01,method='linear')),
        median=None if not finite.size else float(np.median(finite)),
        p99=None if not finite.size else float(np.quantile(finite,.99,method='linear')),
        max=None if not finite.size else float(finite.max()),
        floating_point_quantization_derivative_claim=False)


def representation_metrics(dataset,predicted,normalized,reference,anchors,rows):
    """Descriptive support/physical-error evidence; never used for eligibility."""
    p,z,ref,a=[np.asarray(v,np.float64) for v in (predicted,normalized,reference,anchors)]
    need(p.shape==z.shape==ref.shape==(120,10,7) and a.shape==(120,7) and len(rows)==120
         and all(np.isfinite(v).all() for v in (p,z,ref,a)), 'Complete finite diagnostic arrays required')
    stats=dataset.normalization['residual_actions'];codec.validate_stats(stats,mode=dataset.mode)
    residual=[]
    for value,row in zip(z,rows):
        low,high=codec._range(row['phase'],stats,dataset.mode)
        residual.append((value[:,3]+1)/2*(high[...,3]-low[...,3])+low[...,3])
    r=np.stack(residual);need(np.isfinite(r).all(),'Finite denormalized q4 residual required')
    q=p[:,:,3];zero=q==0;upper=q==codec.U_SAFE_RAD;error=p-ref
    endpoint=(zero|upper)&(r!=0)
    degenerate=((a[:,None,3]==0)&(r<0))|((a[:,None,3]==codec.U_SAFE_RAD)&(r>0))
    per_joint={}
    for name,values,axis in (('first',error[:,0,:],0),('full',error,(0,1))):
        per_joint[name+'_mae_rad']=np.abs(values).mean(axis=axis).tolist()
        per_joint[name+'_bias_rad']=values.mean(axis=axis).tolist()
        per_joint[name+'_max_abs_rad']=np.abs(values).max(axis=axis).tolist()
    displacement={}
    for name,values in (('predicted',p-a[:,None,:]),('reference',ref-a[:,None,:])):
        displacement[name]=dict(first_mean_rad=values[:,0,:].mean(axis=0).tolist(),
            first_mean_abs_rad=np.abs(values[:,0,:]).mean(axis=0).tolist(),
            full_mean_rad=values.mean(axis=(0,1)).tolist(),full_mean_abs_rad=np.abs(values).mean(axis=(0,1)).tolist())
    return dict(descriptive_only=True,affects_eligibility=False,per_joint=per_joint,
        displacement_from_original_measured_anchor=displacement,
        q4=dict(coordinate_count=1200,at_lower_endpoint_count=int(zero.sum()),
            at_upper_endpoint_count=int(upper.sum()),
            nonzero_residual_endpoint_count=int(endpoint.sum()),
            nonzero_residual_degenerate_endpoint_count=int((endpoint&degenerate).sum()),
            nonzero_residual_endpoint_rounding_count=int((endpoint&~degenerate).sum()),
            slope=q4_slopes(r,a[:,None,3])),
        interpretation='q4 support is built into the codec; learned motion and physical success require their original separate gates')


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
    return rows,ref,anchor,metrics(p,ref,anchor,rows),representation_metrics(dataset,p,z,ref,anchor,rows)


def _report(binding,rows,result,numeric_sha,description):
    return dict(schema=SCHEMA,mode=binding['mode'],checkpoint_binding=binding,sampler=SAMPLER,
        rows=rows,diagnostic_schedule_sha256=digest(rows),numeric_sha256=numeric_sha,metrics=result,representation_metrics=description,
        local_step=binding['local_step'],optimizer_step=binding['optimizer_step'],
        initialization_baseline_only=binding['local_step']==0,train_predictions_only=True,
        heldout_predictions_generated=False,closed_loop_validated=False)


def _report_path(binding):
    return Path(binding['checkpoint_path']).parents[1]/'diagnostics'/('local_%06d'%binding['local_step'])/'report.json'


def save_checkpoint_report(output,dataset,predictions,normalized_predictions,checkpoint_binding):
    rows,ref,anchor,result,description=_numeric(dataset,predictions,normalized_predictions,checkpoint_binding)
    output=Path(output).resolve()
    need(output==_report_path(checkpoint_binding).parent.resolve(),'Original candidate report directory required')
    output.mkdir(parents=True,exist_ok=False);numeric=output/'predictions.npz'
    with numeric.open('xb') as f:
        np.savez_compressed(f,predictions=predictions,normalized_predictions=normalized_predictions,reference=ref,anchors=anchor)
        f.flush();os.fsync(f.fileno())
    verify_checkpoint_binding(checkpoint_binding)
    report=_report(checkpoint_binding,rows,result,file_sha(numeric),description);write_new(output/'report.json',report)
    return report


def verify_checkpoint_report(path,dataset,checkpoint_binding):
    path=Path(path);path=path/'report.json' if path.is_dir() else path
    need(path.resolve()==_report_path(checkpoint_binding).resolve(),'Original candidate report path required')
    saved,report_sha=read_pinned(path);numeric=path.with_name('predictions.npz');before=file_sha(numeric)
    need(saved.get('numeric_sha256')==before,'Numeric SHA differs')
    with np.load(numeric,allow_pickle=False) as f:
        need(set(f.files)=={'predictions','normalized_predictions','reference','anchors'},'Numeric schema differs')
        rows,ref,anchor,result,description=_numeric(dataset,f['predictions'],f['normalized_predictions'],checkpoint_binding)
        need(np.array_equal(f['reference'],ref) and np.array_equal(f['anchors'],anchor),'Original labels/anchors changed')
    expected=_report(checkpoint_binding,rows,result,before,description)
    need(saved==expected and file_sha(numeric)==before and file_sha(path)==report_sha,
         'Report or numeric evidence differs from reconstruction')
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
    return dict(selected=selected,status='train_fit_candidate_requires_comparison_and_fixed160' if selected else 'rejected_no_eligible_candidate',
                descriptive_only_local_step=None,
                service_eligible=False)


def load_context(run):
    from .train import verified_run_context
    run=Path(run).resolve(strict=True)
    identities={str(run/name):file_sha(run/name) for name in CONTEXT_FILES}
    context=verified_run_context(run)
    verify_context_inputs(run,identities)
    verify_terminal_segment(run,context.get('terminal_segment_binding'),identities)
    return dict(context,verified_input_sha256=identities)


def verify_context_inputs(run,identities):
    need(isinstance(identities,dict) and set(identities)=={str(run/name) for name in CONTEXT_FILES},
         'Original pre-context input identities required')
    for path,expected in identities.items():
        need(Path(path).is_file() and not Path(path).is_symlink() and file_sha(path)==expected,
             'Consumed context input changed')


def verify_terminal_segment(run,binding,identities):
    need(isinstance(binding,dict) and set(binding)=={'path','sha256','cancellation_path'},
         'Current terminal segment binding required')
    path=Path(binding['path']);cancel=Path(binding['cancellation_path'])
    need(path.is_absolute() and path.resolve()==path and path.parent==run/'training_segments'
         and path.suffix=='.json' and not path.name.endswith(('.cancelled.json','.previous.json'))
         and path.is_file() and not path.is_symlink(),
         'Original direct current terminal segment required')
    need(cancel==path.with_suffix('.cancelled.json') and not cancel.exists() and not cancel.is_symlink(),
         'Current training segment cancellation marker present')
    value,identity=read_pinned(path,binding['sha256'])
    need(identity==identities[str(run/'training_run.json')],
         'Current terminal segment and original training receipt bytes differ')
    receipt,_=read_pinned(run/'training_run.json',identities[str(run/'training_run.json')])
    need(value==receipt and receipt.get('current_segment_path')==str(path),
         'Current terminal segment differs from verified training receipt')
    need(file_sha(path)==binding['sha256']
         and file_sha(run/'training_run.json')==identities[str(run/'training_run.json')]
         and not cancel.exists() and not cancel.is_symlink(),
         'Current terminal segment changed or was cancelled during verification')
    return identity


def build_selection(context):
    from .train import checkpoint_binding
    config=context['config'];run=Path(config['training_run_path']).resolve(strict=True);data=context['dataset']
    need(context.get('completed') is True and config['mode']==data.mode,'Complete matching-arm training required')
    verify_context_inputs(run,context.get('verified_input_sha256'))
    terminal=context.get('terminal_segment_binding')
    verify_terminal_segment(run,terminal,context['verified_input_sha256'])
    need(tuple(context['schedule']['candidate_steps'])==EXPECTED_STEPS and context['schedule']['total_steps']==14600,
         'Complete declared 14600 steps required')
    paths=[Path(p).resolve(strict=True) for p in context['candidate_paths']]
    need(paths==[run/'model'/('step-%012d'%s) for s in EXPECTED_STEPS],'Original thirteen candidate paths required')
    artifacts=dict(context['verified_input_sha256'])
    artifacts[terminal['path']]=terminal['sha256']
    rows=build_diagnostic_schedule(data)
    need(read(run/'diagnostic_schedule.json')==dict(data_hash=data.data_hash,rows=rows),'Predeclared diagnostic origins changed')
    candidates=[]
    for step,path in zip(EXPECTED_STEPS,paths):
        binding=checkpoint_binding(path)
        need(binding['local_step']==binding['optimizer_step']==step and binding['mode']==data.mode and
             binding['config_sha256']==digest(config) and binding['source_sha256']==config['source_sha256'],
             'Candidate clock, arm or source differs')
        report_path=_report_path(binding)
        identities={str(p):file_sha(p) for p in (report_path,report_path.with_name('predictions.npz'))}
        report=verify_checkpoint_report(report_path,data,binding);m=report['metrics']
        for p,h in identities.items():
            need(file_sha(p)==h,'Report identity changed after verification');artifacts[p]=h
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
    verify_terminal_segment(run,terminal,context['verified_input_sha256'])
    return dict(schema=SELECTION_SCHEMA,run=str(run),mode=data.mode,criterion=RULE,
        source_sha256=config['source_sha256'],candidate_steps=list(EXPECTED_STEPS),candidate_count=13,
        full_budget_complete=True,artifact_sha256=artifacts,diagnostic_schedule_sha256=digest(rows),candidates=candidates,
        **result,heldout_predictions_generated=False,generalization_validated=False,closed_loop_validated=False,
        limitations='Original eight TRAIN only. Completed original B condition, full C budget, fixed160 comparison and physical execution remain required.')


def _marker(path):
    return dict(schema='wam_rational_q4_selection_marker_v1',selection_lock=str(path),selection_lock_sha256=file_sha(path))


def _lock(selection,path):
    return dict(schema=LOCK_SCHEMA,selection_path=str(path),selection_sha256=file_sha(path),selection=selection,
                scope='Frozen TRAIN-fit outcome, including negative outcome. Not service or physical admission.')


def verify_selection_lock(path,expected_sha=None):
    path=Path(path).resolve(strict=True);before=file_sha(path)
    need(expected_sha is None or before==expected_sha,'Lock SHA differs')
    lock=read(path);need(lock.get('schema')==LOCK_SCHEMA,'Paired selection lock required')
    run=Path(lock['selection']['run']).resolve(strict=True)
    marker_path=run/MARKER;marker_sha=file_sha(marker_path)
    need(path==run/'rational_selection/selection.lock.json' and read(marker_path)==_marker(path),'Original run/marker required')
    selection_path=path.with_name('selection.json')
    need(lock['selection_path']==str(selection_path) and lock['selection_sha256']==file_sha(selection_path),'Selection SHA differs')
    rebuilt=build_selection(load_context(run))
    need(rebuilt==read(selection_path)==lock['selection'] and _lock(rebuilt,selection_path)==lock and file_sha(path)==before
         and file_sha(marker_path)==marker_sha,
         'Fresh paired selection reconstruction differs')
    return lock


def select_run(run):
    run=Path(run).resolve(strict=True);context=load_context(run)
    need(context['config']['training_run_path']==str(run),'Copied run cannot select')
    need(not any((run/n).exists() for n in (MARKER,'SELECTION_LOCKED.json','FINAL_TEST_STARTED.json')),'Run already consumed')
    output=run/'rational_selection';need(not output.exists(),'Fresh selection output required')
    write_new(run/MARKER,dict(schema='wam_rational_q4_selection_reserved_v1',run=str(run)))
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
