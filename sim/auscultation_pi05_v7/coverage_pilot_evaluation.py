"""Original-eight TRAIN fit diagnosis. No validation/test prediction or acceptance.

The physical hold baseline repeats the measured observation q, not normalized
zero. Raw decoded proposals remain unchanged, including every bound violation.
"""
import json
import os
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v6.action_contract import LOWER, UPPER
from .action_encoding import decode_chunk
from .coverage_pilot_schedule import PHASES, digest, file_sha, json_bytes, source_hashes

SCHEMA = 'wam_coverage_pilot_train_numeric_report_v1'
BINDING_SCHEMA = 'wam_coverage_pilot_checkpoint_binding_v1'
MOVING_RMS_RAD = 2e-6
SAMPLER = dict(seed=917, integration_steps=10, batch_size=1, training_rng_consumed=False)
TARGETS = frozenset(s+str(i) for s in ('R', 'L') for i in range(1, 5))


def need(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('xb') as f:
        f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())


def build_diagnostic_schedule(dataset):
    """First/middle/last original valid windows in each of 40 train buckets."""
    episodes = [(i,e) for i,e in enumerate(dataset.episodes) if e['split']=='train']
    ids = [e['episode_id'] for _,e in episodes]
    need(len(ids)==len(set(ids))==8 and ids==dataset.normalization['residual_actions']['fit_episode_ids'],
         'Exactly original eight train normalization identities required')
    grouped = {(eid,p):[] for eid in ids for p in PHASES}; origins=set(); targets={}
    for ti, (ei,start) in enumerate(dataset.indices['train']):
        need(isinstance(ei,(int,np.integer)) and isinstance(start,(int,np.integer)) and
             not isinstance(ei,(bool,np.bool_)) and not isinstance(start,(bool,np.bool_)) and
             0<=ei<len(dataset.episodes) and start>=0, 'Invalid train origin')
        ep=dataset.episodes[ei]; need(ep['split']=='train' and ep['episode_id'] in ids, 'Heldout train index')
        phase=str(ep['phases'][start]); target=str(ep['targets'][start])
        need(phase in PHASES and target in TARGETS, 'Unknown original train phase/target')
        need(len(ep['actions'][start:start+10])==10 and
             all(str(x)==phase for x in ep['phases'][start:start+10]) and
             all(str(x)==target for x in ep['targets'][start:start+10]), 'Incomplete/cross-boundary window')
        need((ei,start) not in origins, 'Duplicate qualified origin'); origins.add((ei,start))
        need(targets.setdefault(ep['episode_id'],target)==target, 'Mixed episode target')
        grouped[ep['episode_id'],phase].append(dict(episode_id=ep['episode_id'],target=target,phase=phase,
            episode_index=int(ei),train_index=ti,start=int(start)))
    need(set(targets.values())==TARGETS and len(targets)==8, 'All original eight target identities required')
    rows=[]
    for eid in ids:
        for phase in PHASES:
            bucket=sorted(grouped[eid,phase],key=lambda r:r['start'])
            need(len(bucket)>=3, 'At least three unique qualified origins per train phase required')
            for position,index in zip(('first','middle','last'),(0,len(bucket)//2,len(bucket)-1)):
                rows.append(dict(bucket[index],row_id=len(rows),position=position))
    return rows


def references(dataset, rows):
    need(rows==build_diagnostic_schedule(dataset), 'Diagnostic schedule differs from original qualified train indices')
    ref=[]; anchor=[]
    for row in rows:
        ep=dataset.episodes[row['episode_index']]; start=row['start']
        ref.append(ep['actions'][start:start+10]); anchor.append(ep['state'][start,:7])
    ref,anchor=map(np.asarray,(ref,anchor))
    need(ref.shape==(120,10,7) and anchor.shape==(120,7) and
         np.isfinite(ref).all() and np.isfinite(anchor).all(), 'Nonfinite/malformed raw labels/anchors')
    return ref,anchor


def _shape_rows(rows):
    need(len(rows)==120 and [r['row_id'] for r in rows]==list(range(120)), 'Fixed 120-row denominator required')
    keys=[(r['episode_id'],r['phase'],r['position']) for r in rows]
    need(len(set(keys))==120 and len({r['episode_id'] for r in rows})==8 and
         {r['target'] for r in rows}==TARGETS, 'Eight unique targets and unique origins required')
    for eid in dict.fromkeys(r['episode_id'] for r in rows):
        for phase in PHASES:
            own=[r for r in rows if r['episode_id']==eid and r['phase']==phase]
            need(len(own)==3 and [r['position'] for r in own]==['first','middle','last'] and
                 len({r['start'] for r in own})==3, 'Three distinct phase origins required')


def bounds_report(prediction):
    p=np.asarray(prediction,dtype=np.float64); need(p.shape==(120,10,7) and np.isfinite(p).all(), 'Finite 120x10x7 raw proposals required')
    invalid=(p<np.asarray(LOWER))|(p>np.asarray(UPPER))
    return dict(all_passed=not bool(invalid.any()), full_chunk_bound_pass_fraction=float(np.mean(~invalid.any(axis=(1,2)))),
        first_action_bound_pass_fraction=float(np.mean(~invalid[:,0].any(axis=1))),coordinate_count=int(p.size),
        violating_coordinates=int(invalid.sum()),violations=[dict(row_id=int(i),horizon_index=int(h),joint_index=int(j),
        proposed_rad=float(p[i,h,j])) for i,h,j in np.argwhere(invalid)],
        per_joint_min_margin_rad=np.minimum(p-np.asarray(LOWER),np.asarray(UPPER)-p).min(axis=(0,1)).tolist(),
        lower_rad=list(LOWER),upper_rad=list(UPPER),tolerance_rad=0.,clipping=False)


def metrics(predictions, reference, anchors, rows):
    _shape_rows(rows)
    p,r,a=map(lambda x:np.asarray(x,dtype=np.float64),(predictions,reference,anchors))
    need(p.shape==r.shape==(120,10,7) and a.shape==(120,7) and
         all(np.isfinite(x).all() for x in (p,r,a)), 'Invalid numeric arrays')
    delta=r-a[:,None,:]; proposed=p-a[:,None,:]; err=p-r
    rms=np.sqrt(np.mean(delta**2,axis=(1,2))); moving=rms>MOVING_RMS_RAD
    buckets=[]; initial=[]
    for eid in dict.fromkeys(row['episode_id'] for row in rows):
        for phase in PHASES:
            ix=np.array([i for i,row in enumerate(rows) if row['episode_id']==eid and row['phase']==phase])
            mi=ix[moving[ix]]; movement_required=phase in ('FREE','APPROACH')
            movement=dict(count=len(mi),status='unassessable_no_moving_origin',passed=False,
                full_mae_rad=None,hold_full_mae_rad=None,projection_progress=None)
            if len(mi):
                mae=float(np.abs(err[mi]).mean()); hold=float(np.abs(delta[mi]).mean())
                progress=float(np.sum(proposed[mi]*delta[mi])/np.sum(delta[mi]**2))
                passed=mae<hold and progress>0
                movement.update(status='passed' if passed else 'failed',passed=passed,
                    full_mae_rad=mae,hold_full_mae_rad=hold,projection_progress=progress)
            b=dict(episode_id=eid,target=rows[int(ix[0])]['target'],phase=phase,count=len(ix),
                near_zero_count=int((~moving[ix]).sum()),moving_required=movement_required,moving=movement,
                first_mae_rad=float(np.abs(err[ix,0]).mean()),full_mae_rad=float(np.abs(err[ix]).mean()),
                rmse_rad=float(np.sqrt(np.mean(err[ix]**2))),max_abs_error_rad=float(np.abs(err[ix]).max()),
                first_hold_mae_rad=float(np.abs(delta[ix,0]).mean()),full_hold_mae_rad=float(np.abs(delta[ix]).mean()),
                first_joint_bias_rad=err[ix,0].mean(axis=0).tolist())
            buckets.append(b)
            if phase=='FREE':
                i=int(ix[0]); need(rows[i]['position']=='first','Initial FREE origin mismatch')
                initial.append(dict(episode_id=eid,target=rows[i]['target'],row_id=i,
                    max_abs_error_rad=float(np.abs(err[i,0]).max()),first_mae_rad=float(np.abs(err[i,0]).mean()),
                    hold_first_mae_rad=float(np.abs(delta[i,0]).mean()),reference_delta_rad=delta[i,0].tolist(),
                    predicted_delta_rad=proposed[i,0].tolist(),signed_error_rad=err[i,0].tolist(),
                    reference_motion_rms_rad=float(rms[i]),near_zero=bool(not moving[i])))
    required=[b for b in buckets if b['moving_required']]; passed=sum(b['moving']['passed'] for b in required)
    bounds=bounds_report(p); eligible=bool(bounds['all_passed'] and passed==16)
    return dict(eligible=eligible,bounds=bounds,moving_buckets=dict(required=16,passed=passed,
        unassessable=sum(b['moving']['status']=='unassessable_no_moving_origin' for b in required)),
        scores=dict(worst_initial_first_max_abs_rad=max(v['max_abs_error_rad'] for v in initial),
            macro_first_mae_rad=float(np.mean([b['first_mae_rad'] for b in buckets])),
            macro_full_mae_rad=float(np.mean([b['full_mae_rad'] for b in buckets]))),
        initial_free=initial,buckets=buckets,origins=[dict(row_id=i,moving=bool(moving[i]),
            reference_delta_rms_rad=float(rms[i]),first_mae_rad=float(np.abs(err[i,0]).mean()),
            full_mae_rad=float(np.abs(err[i]).mean()),first_hold_mae_rad=float(np.abs(delta[i,0]).mean()),
            full_hold_mae_rad=float(np.abs(delta[i]).mean())) for i in range(120)],
        definitions=dict(moving_reference_full_rms_rad_strictly_greater=MOVING_RMS_RAD,
            moving_gate='each FREE/APPROACH bucket: moving-subset full MAE strictly below hold and pooled projection > 0',
            projection='sum((prediction-anchor)*(reference-anchor))/sum((reference-anchor)^2), pooled over moving origins',
            hold='repeat raw measured observation q for every horizon step; not normalized zero',
            ranking='worst eight initial first maxabs, equal40bucket first MAE, full MAE, earlier local step',
            tiny_joint_sign_gate=False,near_zero_origins_reported=True),
        scope='original-eight train-fit diagnosis only; no generalization or closed-loop acceptance')


def verify_checkpoint_binding(binding):
    from .coverage_pilot_train import checkpoint_binding
    keys={'schema','checkpoint_path','local_step','optimizer_step','metadata_sha256','arrays_sha256',
          'complete_sha256','data_hash','normalization_sha256','config_sha256','schedule_sha256',
          'pilot_binding','pilot_source_sha256'}
    need(set(binding)==keys,'Exact pilot checkpoint binding fields required')
    need(binding.get('schema')==BINDING_SCHEMA,'Pilot checkpoint schema required')
    need(type(binding['local_step']) is int and binding['local_step']>=0 and
         binding['optimizer_step']==4000+binding['local_step'],'Pilot/ancestral optimizer clocks differ')
    need(binding['pilot_source_sha256']==source_hashes(),'Pilot source bytes changed')
    need(checkpoint_binding(binding['checkpoint_path'])==binding,'Checkpoint bytes/binding changed')


def _numeric(dataset, predicted, normalized, binding):
    verify_checkpoint_binding(binding)
    need(binding['data_hash']==dataset.data_hash and binding['normalization_sha256']==digest(dataset.normalization), 'Dataset/normalization binding changed')
    rows=build_diagnostic_schedule(dataset);ref,anchor=references(dataset,rows)
    p,z=map(np.asarray,(predicted,normalized))
    need(p.shape==z.shape==(120,10,7) and p.dtype.kind==z.dtype.kind=='f' and
         np.isfinite(p).all() and np.isfinite(z).all(), 'Finite decoded and normalized arrays required')
    decoded=np.stack([decode_chunk(v,a,row['phase'],dataset.normalization['residual_actions']) for v,a,row in zip(z,anchor,rows)])
    need(np.array_equal(p,decoded),'Raw proposal differs from exact frozen codec decode')
    return rows,ref,anchor,metrics(p,ref,anchor,rows)


def _report(binding,rows,result,numeric_sha):
    return dict(schema=SCHEMA,checkpoint_binding=binding,sampler=SAMPLER,rows=rows,
        diagnostic_schedule_sha256=digest(rows),numeric_sha256=numeric_sha,metrics=result,
        local_step=binding['local_step'],train_predictions_only=True,heldout_predictions_generated=False,
        closed_loop_validated=False,initialization_baseline_only=binding['local_step']==0)


def save_checkpoint_report(output,dataset,predictions,normalized_predictions,checkpoint_binding):
    rows,ref,anchor,result=_numeric(dataset,predictions,normalized_predictions,checkpoint_binding)
    output=Path(output).resolve(); checkpoint=Path(checkpoint_binding['checkpoint_path'])
    expected=checkpoint.parents[1]/'diagnostics'/('local_%06d'%checkpoint_binding['local_step'])
    need(output==expected.resolve(),'Diagnostic output must be the candidate-specific original run directory')
    output.mkdir(parents=True,exist_ok=False)
    numeric=output/'predictions.npz'
    with numeric.open('xb') as f:
        np.savez_compressed(f,predictions=np.asarray(predictions),normalized_predictions=np.asarray(normalized_predictions),reference=ref,anchors=anchor)
        f.flush();os.fsync(f.fileno())
    verify_checkpoint_binding(checkpoint_binding)
    report=_report(checkpoint_binding,rows,result,file_sha(numeric));write_new(output/'report.json',report)
    return report


def verify_checkpoint_report(path,dataset,checkpoint_binding):
    path=Path(path);path=path/'report.json' if path.is_dir() else path
    expected=Path(checkpoint_binding['checkpoint_path']).parents[1]/'diagnostics'/('local_%06d'%checkpoint_binding['local_step'])/'report.json'
    need(path.resolve()==expected.resolve(),'Original candidate diagnostic path required')
    saved=read(path);numeric=path.with_name('predictions.npz');before=file_sha(numeric)
    need(saved.get('numeric_sha256')==before,'Numeric artifact SHA mismatch')
    with np.load(numeric,allow_pickle=False) as f:
        need(set(f.files)=={'predictions','normalized_predictions','reference','anchors'},'Numeric schema differs')
        rows,ref,anchor,result=_numeric(dataset,f['predictions'],f['normalized_predictions'],checkpoint_binding)
        need(np.array_equal(f['reference'],ref) and np.array_equal(f['anchors'],anchor),'Raw source labels/anchors changed')
    expected=_report(checkpoint_binding,rows,result,before)
    need(saved==expected,'Report metrics/source binding differ from fresh numeric reconstruction')
    need(file_sha(numeric)==before,'Numeric evidence changed during verification')
    return expected
