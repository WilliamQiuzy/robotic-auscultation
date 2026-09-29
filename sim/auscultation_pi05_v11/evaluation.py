"""Full R5 evaluation and immutable selection for the declared V11 repair.

Offline joint-envelope checks do not replace FK, collision or contact tests.
Nonfinite model results retain their bytes and are completed negative reports;
missing rows or infrastructure failures never become invented negative results.
"""
from pathlib import Path
import json
import os
import numpy as np
from sim.auscultation_pi05_v6.action_contract import LOWER, UPPER
from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as original
from sim.auscultation_pi05_v7.coverage_pilot_schedule import file_sha, digest, json_bytes
from . import training_data as D

PHASES=('FREE','APPROACH','CONTACT','HOLD','RETRACT')
STEPS=tuple(range(0,4001,500))
SAMPLER=dict(seed=917,integration_steps=10,batch_size=1,training_rng_consumed=False)
SCHEMA='wam_pi05_v11_bounded_numeric_report_v1'
LOCK_SCHEMA='wam_pi05_v11_bounded_r5_selection_lock_v1'
MARKER='V11_SELECTION_LOCKED.json'
need=original.need
read=original.read
write_new=original.write_new


def build_report_schedule(dataset,split):
    need(split in ('train','validation','test') and split in dataset.loaded_scopes,'Qualified requested split required')
    rows=[]
    if split=='train':
        view=D.original_train_view(dataset)
        old=original.build_diagnostic_schedule(view)
        lookup={(dataset.episodes[ei]['episode_id'],int(s)):i for i,(ei,s) in enumerate(dataset.indices['train'])}
        eis={e['episode_id']:i for i,e in enumerate(dataset.episodes)}
        for r in old:
            rows.append(dict(r,split='train',group='original120',source_train_index=r['train_index'],source_episode_index=r['episode_index'],
                train_index=lookup[r['episode_id'],r['start']],episode_index=eis[r['episode_id']]))
        original_ids=set(D.original_ids(dataset))
        for ei,e in enumerate(dataset.episodes):
            if e['split']!='train' or e['episode_id'] in original_ids:continue
            for phase in PHASES:
                cell=[(ti,int(s)) for ti,(j,s) in enumerate(dataset.indices['train']) if j==ei and str(e['phases'][s])==phase]
                cell.sort(key=lambda r:r[1])
                if not cell:continue
                for index in sorted({0,len(cell)//2,len(cell)-1}):
                    ti,start=cell[index]
                    rows.append(dict(row_id=len(rows),split=split,group='added_train',episode_id=e['episode_id'],episode_index=ei,
                        train_index=ti,start=start,phase=phase,target=str(e['targets'][start]),position='first' if index==0 else 'last' if index==len(cell)-1 else 'middle'))
    else:
        for phase in PHASES:
            cell=[]
            for si,(ei,start) in enumerate(dataset.indices[split]):
                e=dataset.episodes[ei]
                if str(e['phases'][start])==phase:cell.append(dict(split=split,group=split,episode_id=e['episode_id'],episode_index=int(ei),
                    split_index=si,start=int(start),phase=phase,target=str(e['targets'][start])))
            need(cell,'Every heldout phase must have qualified windows')
            for r in sorted(cell,key=lambda r:(r['episode_id'],r['start'])):rows.append(dict(r,row_id=len(rows)))
        need(len(rows)==len(dataset.indices[split]),'Complete heldout denominator required')
    need(rows and len({(r['episode_id'],r['start']) for r in rows})==len(rows),'Unique nonempty report origins required')
    return rows


def build_probe_schedule(dataset):
    train=dataset.indices['train'];rows=[];original_ids=D.original_ids(dataset)
    for ids,phases,group in ((original_ids,('FREE',),'original160'),
        ([e['episode_id'] for e in dataset.episodes if e['split']=='train' and e['episode_id'] not in original_ids],PHASES[:2],'added_precontact')):
        for eid in ids:
            ei=next(i for i,e in enumerate(dataset.episodes) if e['episode_id']==eid);e=dataset.episodes[ei]
            for phase in phases:
                cell=sorted([(ti,int(s)) for ti,(j,s) in enumerate(train) if j==ei and str(e['phases'][s])==phase],key=lambda x:x[1])
                if group=='original160':need(len(cell)>=20,'Original20 FREE rows per episode required')
                for ti,start in cell[:20]:rows.append(dict(row_id=len(rows),split='train',group=group,episode_id=eid,episode_index=ei,train_index=ti,start=start,phase=phase,target=str(e['targets'][start])))
    need(sum(r['group']=='original160' for r in rows)==160,'Original160 denominator differs')
    return rows


def references(dataset,rows):
    ref=[];anchor=[]
    for r in rows:
        e=dataset.episodes[r['episode_index']];s=r['start']
        need(e['episode_id']==r['episode_id'] and e['split']==r['split'] and e['split'] in dataset.loaded_scopes,'Origin outside qualified split')
        need(all(str(x)==r['phase'] for x in e['phases'][s:s+10]),'Cross-phase origin')
        need(e['target']==r['target'] and all(str(x)==r['target'] for x in e['targets'][s:s+10]),'Reference target mismatch')
        # References are audited here, never encoded while constructing input.
        D.codec.inverse_coordinates(e['actions'][s:s+10],r['target'],dataset.normalization['residual_actions']['bounds_contract'])
        ref.append(e['actions'][s:s+10]);anchor.append(e['state'][s,:7])
    ref,anchor=map(np.asarray,(ref,anchor))
    need(ref.shape==(len(rows),10,7) and anchor.shape==(len(rows),7) and np.isfinite(ref).all() and np.isfinite(anchor).all(),'Malformed qualified references')
    return ref,anchor


def posture_contract():
    from .train import source_hashes
    sources=source_hashes();root=Path(__file__).resolve().parents[2];plans={};bindings={}
    for target in ('R1','L1','R2','L2','R3','L3','R4','L4','R5','L5'):
        rel='sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_'+target+'.json'
        p=root/rel;need(file_sha(p)==sources[rel],'Frozen physical posture plan changed')
        plans[target]=read(p)['points'][0]['posture_limits'];bindings[rel]=sources[rel]
    return plans,bindings


def raw_gates(predicted,rows,plans):
    p=np.asarray(predicted,dtype=np.float64)
    need(p.shape==(len(rows),10,7),'All70 proposals required')
    finite=np.isfinite(p);low=np.asarray([plans[r['target']]['joint_lower_rad'] for r in rows])[:,None,:]
    high=np.asarray([plans[r['target']]['joint_upper_rad'] for r in rows])[:,None,:]
    urdf=(p<LOWER)|(p>UPPER);posture=np.minimum(p-low,high-p)<-1e-9
    def summary(mask,margin,tolerance):
        return dict(all_passed=bool(finite.all() and not mask.any()),violating_coordinates=int(mask.sum()),nonfinite_coordinates=int((~finite).sum()),
            coordinate_count=int(p.size),tolerance_rad=tolerance,min_margin_rad=float(margin.min()) if finite.all() else None,
            violations=[dict(row_id=int(i),horizon_index=int(h),joint_index=int(j),proposed_rad=float(p[i,h,j]) if finite[i,h,j] else None,
                numeric_status='finite' if finite[i,h,j] else 'nonfinite_retained_in_npz') for i,h,j in np.argwhere(mask)],clipping=False)
    return dict(urdf=summary(urdf,np.minimum(p-LOWER,UPPER-p),0.),posture=summary(posture,np.minimum(p-low,high-p),1e-9),
        scope='Original command joint-envelope predicate only; FK/elbow/collision/contact require Isaac',observed_tolerance_used=False)


def phase_metrics(predictions,reference,anchors,rows):
    p,r,a=map(lambda v:np.asarray(v,dtype=np.float64),(predictions,reference,anchors))
    need(p.shape==r.shape==(len(rows),10,7) and a.shape==(len(rows),7),'Numeric shapes differ')
    if not np.isfinite(p).all():return dict(rankable=False,macro_first_mae_rad=None,macro_full_mae_rad=None,phases=[])
    err=p-r;delta=r-a[:,None,:];prop=p-a[:,None,:];phases=[]
    for phase in PHASES:
        ix=np.array([i for i,row in enumerate(rows) if row['phase']==phase],dtype=int)
        if not len(ix):continue
        moving=ix[np.sqrt(np.mean(delta[ix]**2,axis=(1,2)))>2e-6]
        m=dict(count=int(len(moving)),status='unassessable_no_moving_origin',projection_progress=None,mae_below_hold=None)
        if len(moving):m.update(status='descriptive',projection_progress=float(np.sum(prop[moving]*delta[moving])/np.sum(delta[moving]**2)),mae_below_hold=bool(abs(err[moving]).mean()<abs(delta[moving]).mean()))
        phases.append(dict(phase=phase,count=int(len(ix)),first_mae_rad=float(abs(err[ix,0]).mean()),full_mae_rad=float(abs(err[ix]).mean()),
            first_joint_bias_rad=err[ix,0].mean(0).tolist(),full_joint_bias_rad=err[ix].mean((0,1)).tolist(),
            first_joint_mae_rad=abs(err[ix,0]).mean(0).tolist(),full_joint_mae_rad=abs(err[ix]).mean((0,1)).tolist(),
            max_abs_rad=float(abs(err[ix]).max()),first_hold_mae_rad=float(abs(delta[ix,0]).mean()),full_hold_mae_rad=float(abs(delta[ix]).mean()),moving=m))
    need(phases,'Nonempty phases required')
    return dict(rankable=True,macro_first_mae_rad=float(np.mean([x['first_mae_rad'] for x in phases])),macro_full_mae_rad=float(np.mean([x['full_mae_rad'] for x in phases])),phases=phases)


def decode_accounted(dataset,normalized,rows):
    z=np.asarray(normalized);need(z.shape==(len(rows),10,7) and z.dtype.kind=='f','Complete real model output tensor required')
    result=[];statuses=[]
    for v,r in zip(z,rows):
        if not np.isfinite(v).all():
            result.append(np.full((10,7),np.nan,np.float32));statuses.append('raw_unavailable_nonfinite_normalized');continue
        e=dataset.episodes[r['episode_index']]
        need(e['episode_id']==r['episode_id'] and str(e['targets'][r['start']])==r['target'] and str(e['phases'][r['start']])==r['phase'],'Decoded row target/phase mismatch')
        # Identity/source/decoder exceptions are infrastructure failures. Only
        # actual nonfinite returned numeric values form a negative model result.
        raw=D.decode_prediction(dataset,v,e['state'][r['start']],r['phase'],r['target'])
        need(np.asarray(raw).shape==(10,7),'Decoder output shape differs')
        result.append(raw);statuses.append('finite_decoded' if np.isfinite(raw).all() else 'nonfinite_decoded')
    return np.asarray(result),statuses


def coordinate_diagnostics(dataset,predicted,rows):
    p=np.asarray(predicted,dtype=np.float64);bounds=dataset.normalization['residual_actions']['bounds_contract']
    low=np.asarray([bounds['targets'][r['target']]['lower_rad'] for r in rows])[:,None,:]
    high=np.asarray([bounds['targets'][r['target']]['upper_rad'] for r in rows])[:,None,:]
    finite=np.isfinite(p);margin=np.minimum(p-low,high-p)
    return dict(coordinate_count=int(p.size),finite_coordinates=int(finite.sum()),
        endpoint_saturated_coordinates=int((finite & ((p==low)|(p==high))).sum()),
        within_one_tau_of_endpoint_coordinates=int((finite & (margin<=.001)).sum()),tau_rad=.001,
        scope='Descriptive architectural support and saturation; not learned safety or physical acceptance')


def numerical_report(dataset,normalized,rows,plans):
    p,status=decode_accounted(dataset,normalized,rows);ref,a=references(dataset,rows)
    gates=raw_gates(p,rows,plans);metrics=phase_metrics(p,ref,a,rows)
    coordinate=coordinate_diagnostics(dataset,p,rows)
    result=dict(count=len(rows),row_status=status,finite_all=all(x=='finite_decoded' for x in status),raw_gates=gates,metrics=metrics,
        coordinate_diagnostics=coordinate,normalized_outside_unit_coordinate_count=int((abs(normalized)>1).sum()),normalized_coordinate_count=int(np.asarray(normalized).size),
        bounds_conformance_is_architectural=True,action_clipping=False,missing_or_infrastructure_rows=0,unavailable_raw_storage='NaN sentinel only for rows explicitly raw_unavailable; never claimed decoded output')
    if rows[0]['split']=='train':
        oldrows=original.build_diagnostic_schedule(D.original_train_view(dataset))
        need(len(rows)>=120 and all(r['group']=='original120' for r in rows[:120]),'Original120 first required')
        result['original120']=original.metrics(p[:120],ref[:120],a[:120],oldrows) if np.isfinite(p[:120]).all() else dict(eligible=False,moving_buckets=dict(required=16,passed=0,status='nonfinite_unassessable'))
    return p,ref,a,result


def save_checkpoint_report(output,dataset,normalized_by_split,binding):
    from .train import checkpoint_binding
    need(checkpoint_binding(binding['checkpoint_path'])==binding and binding['data_hash']==dataset.data_hash,'Actual checkpoint binding differs')
    need(set(normalized_by_split)=={'train','validation'},'Both TRAIN diagnostics and full R5 required')
    output=Path(output);expected=Path(binding['checkpoint_path']).parents[1]/'diagnostics'/('local_%06d'%binding['local_step'])
    need(output==expected,'Original report directory required');output.mkdir(parents=True,exist_ok=False)
    plans,pb=posture_contract();groups={}
    for split in ('train','validation'):
        rows=build_report_schedule(dataset,split);z=np.asarray(normalized_by_split[split]);p,ref,a,m=numerical_report(dataset,z,rows,plans)
        numeric=output/(split+'.npz')
        with numeric.open('xb') as f:np.savez_compressed(f,normalized=z,predictions=p,reference=ref,anchors=a);f.flush();os.fsync(f.fileno())
        groups[split]=dict(rows=rows,row_digest=digest(rows),numeric_sha256=file_sha(numeric),result=m)
    report=_assemble(binding,dataset,groups,pb);write_new(output/'report.json',report)
    return report


def _assemble(binding,dataset,groups,posture):
    eligible=all(g['result']['finite_all'] and all(g['result']['raw_gates'][k]['all_passed'] for k in ('urdf','posture')) for g in groups.values())
    eligible=bool(eligible and groups['train']['result']['original120']['moving_buckets']['passed']==16 and binding['local_step']>0)
    val=groups['validation']['result']['metrics']
    return dict(schema=SCHEMA,checkpoint_binding=binding,dataset_binding=D.dataset_binding(dataset),sampler=SAMPLER,groups=groups,
        posture_plan_sha256=posture,eligible=eligible,scores={k:val[k] for k in ('macro_first_mae_rad','macro_full_mae_rad')},
        report_complete=True,step0_baseline_only=binding['local_step']==0,physical_acceptance=False,generalization_validated=False)


def verify_checkpoint_report(path,dataset,binding):
    from .train import checkpoint_binding
    need(checkpoint_binding(binding['checkpoint_path'])==binding,'Checkpoint changed')
    path=Path(path);path=path/'report.json' if path.is_dir() else path
    expected=Path(binding['checkpoint_path']).parents[1]/'diagnostics'/('local_%06d'%binding['local_step'])/'report.json'
    need(path==expected and not path.is_symlink(),'Original report required');before=file_sha(path);saved=read(path)
    plans,pb=posture_contract();groups={}
    for split in ('train','validation'):
        rows=build_report_schedule(dataset,split);numeric=path.with_name(split+'.npz');nh=file_sha(numeric)
        need(nh==saved['groups'][split]['numeric_sha256'],'Numeric bytes differ')
        with np.load(numeric,allow_pickle=False) as f:
            need(set(f.files)=={'normalized','predictions','reference','anchors'},'Numeric members differ')
            p,ref,a,m=numerical_report(dataset,f['normalized'],rows,plans)
            for name,value in (('predictions',p),('reference',ref),('anchors',a)):
                need(f[name].dtype==value.dtype and f[name].tobytes()==value.tobytes(),'Actual numeric bytes differ: '+name)
        groups[split]=dict(rows=rows,row_digest=digest(rows),numeric_sha256=nh,result=m)
        need(file_sha(numeric)==nh,'Numeric changed during verification')
    need(_assemble(binding,dataset,groups,pb)==saved and file_sha(path)==before,'Report reconstruction differs')
    return saved


def choose_candidate(candidates):
    need([r['local_step'] for r in candidates]==list(STEPS),'All nine reports including step0 required')
    usable=[]
    for c in candidates:
        need(type(c['eligible']) is bool,'Boolean eligibility required')
        if c['eligible'] and c['local_step']>0:
            values=[c['scores'][k] for k in ('macro_first_mae_rad','macro_full_mae_rad')]
            need(all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in values),'Finite rankable score required')
            usable.append(c)
    return min(usable,key=lambda c:(c['scores']['macro_first_mae_rad'],c['scores']['macro_full_mae_rad'],c['local_step'])) if usable else None


def build_selection(context):
    from .train import checkpoint_binding
    from .execution import verify_training_owner
    need(context['completed'] is True,'Full4000 actual run required');data=context['dataset'];run=Path(context['config']['training_run_path'])
    artifacts=dict(context['artifact_sha256'])
    owner=verify_training_owner(run,context);artifacts.update(owner['artifact_sha256'])
    candidates=[]
    for step in STEPS:
        path=run/'model'/('step-%012d'%step);b=checkpoint_binding(path);rp=run/'diagnostics'/('local_%06d'%step)/'report.json'
        for p in (rp,rp.with_name('train.npz'),rp.with_name('validation.npz')):artifacts[str(p)]=file_sha(p)
        r=verify_checkpoint_report(rp,data,b)
        candidates.append(dict(local_step=step,checkpoint_path=str(path),checkpoint_binding=b,eligible=r['eligible'],scores=r['scores'],report_path=str(rp),report_sha256=artifacts[str(rp)]))
    selected=choose_candidate(candidates)
    result=dict(schema='wam_pi05_v11_bounded_r5_selection_v1',run=str(run),dataset_binding=D.dataset_binding(data),candidate_count=9,positive_candidate_count=8,
        full_budget_complete=True,artifact_sha256=artifacts,candidates=candidates,selected=selected,
        status='selected_requires_probes_test_and_physics' if selected else 'rejected_no_eligible_candidate',
        rule='full R5 equal-five-phase first MAE, full MAE, earlier positive step; raw safety and original16 movement eligibility',
        test_used_for_selection=False,physical_acceptance=False,population_generalization=False)
    for p,h in artifacts.items():need(file_sha(p)==h,'Selection evidence changed')
    from .train import verify_terminal
    need(verify_terminal(run)==context['terminal'],'Current terminal segment changed during selection')
    need(verify_training_owner(run,context)==owner,'Training owner changed during selection')
    return result


def verify_selection_lock(path,expected_sha=None):
    from .train import verified_run_context
    path=Path(path);need(path.is_absolute() and path==path.resolve() and path.is_file() and not path.is_symlink(),'Canonical lock required')
    before=file_sha(path);need(expected_sha is None or before==expected_sha,'Lock SHA differs');artifacts={};lock=D._read_pinned(path,before,artifacts)
    need(lock.get('schema')==LOCK_SCHEMA,'V11 lock required');run=Path(lock['selection']['run'])
    need(path==run/'selection/selection.lock.json','Original lock path required')
    sp=path.with_name('selection.json');need(lock['selection_path']==str(sp),'Selection path differs')
    saved=D._read_pinned(sp,lock['selection_sha256'],artifacts)
    marker=run/MARKER;marker_value=D._read_pinned(marker,file_sha(marker),artifacts)
    need(marker_value==dict(selection_lock=str(path),selection_lock_sha256=before),'Selection marker differs')
    rebuilt=build_selection(verified_run_context(run))
    need(rebuilt==lock['selection']==saved,'Selection reconstruction differs')
    D._recheck(artifacts)
    return lock


def select_run(run):
    from .train import verified_run_context
    run=Path(run).resolve(strict=True)
    need(not any((run/n).exists() or (run/n).is_symlink() for n in (MARKER,'selection','FINAL_TEST_STARTED.json')),'Run already consumed')
    context=verified_run_context(run);write_new(run/MARKER,dict(status='reserved'))
    selection=build_selection(context);dest=run/'selection';dest.mkdir();sp=dest/'selection.json';write_new(sp,selection)
    lp=dest/'selection.lock.json';write_new(lp,dict(schema=LOCK_SCHEMA,selection_path=str(sp),selection_sha256=file_sha(sp),selection=selection));lp.chmod(0o444)
    tmp=run/(MARKER+'.completed.tmp');write_new(tmp,dict(selection_lock=str(lp),selection_lock_sha256=file_sha(lp)));os.replace(tmp,run/MARKER)
    return dict(selection_lock=str(lp),sha256=file_sha(lp),status=selection['status'],selected_step=None if selection['selected'] is None else selection['selected']['local_step'])
