"""Versioned real-prefix ACT export following the original physical decoder.

No simulator or model is invoked. Histories are always two real observations;
only actually executed complete teacher raw setter cells supply future labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import traceback

import numpy as np

from . import m2_given_contact_act as given, m2_act as shared
from . import m2_given_contact_dataset as data, m2_given_contact_runtime as rt
from . import m2_given_contact_recovery_dataset as recovery
from . import m2_rebuild_capture_dataset as rebuild
from . import m2_given_contact_expanded_pilot_train as trainer

need=rt.need
REQUEST_SCHEMA='m2_given_contact_masked_export_request_v1'
ARRAY_KEYS=shared.ARRAY_KEYS+('action_is_pad',)


def _decode_job(value):
    row,base=value
    try:return row,read_episode(row,base),None
    except Exception as error:return row,None,type(error).__name__+': '+str(error)


def decode_rows(rows,base,*,cpu_workers=1):
    """Same complete physical/fidelity decoder, in source order, with no CUDA."""
    need(type(cpu_workers) is int and 1<=cpu_workers<=3,'One through three CPU decoder workers required')
    jobs=((row,Path(base)) for row in rows)
    if cpu_workers==1:
        yield from map(_decode_job,jobs)
    else:
        # Spawn independent readers; no inherited native/BLAS process state and
        # no shared mutable decoder. map preserves the declared source order.
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            yield from pool.map(_decode_job,jobs)


def read_episode(row,base):
    """Requalify originals before rereading immutable bytes for new labels."""
    config=rt.read_bound(row['config'])
    is_recovery='m2_given_contact_recovery' in config
    samples,receipt,meta=(recovery.read_episode(row,base,allow_truncated=True) if is_recovery else
        data.read_episode(row,base,allow_truncated=True))
    receipt.update(windows=len(samples['states']),label_protocol=shared.MASKED_LABEL_PROTOCOL,
        origin='qualified_actual_recovery' if is_recovery else 'qualified_actual_nominal')
    return samples,receipt,meta


def combine(decoded):
    parts={s:[] for s in ('train','validation')};offsets={s:0 for s in parts};episodes=[];seen=set();resets=set()
    index=dict(schema=data.INDEX_SCHEMA,task_id=data.pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        label_protocol=shared.MASKED_LABEL_PROTOCOL,episodes=[],samples=[])
    refs=[]
    for row,(sample,receipt,meta) in decoded:
        given.require_samples(sample)
        need(sample['schema']==given.MASKED_DATASET_SCHEMA,'Only explicitly masked real-prefix samples')
        split=row['split'];shared.require_membership(split,row['subject_id'])
        eid=row['episode_id'];need(eid not in seen,'Duplicate positive episode');seen.add(eid)
        identity=receipt.get('native_reset_identity',{})
        need(type(identity.get('pid')) is int and type(identity.get('sid')) is int and
             isinstance(identity.get('start_ticks'),str),'Native reset identity required')
        reset=(identity['pid'],identity['sid'],identity['start_ticks'])
        need(reset not in resets,'Duplicate native process reused across episode resets');resets.add(reset)
        count=len(sample['states']);need(count>0,'Positive real-prefix windows required')
        start=offsets[split];offsets[split]+=count;parts[split].append(sample)
        episode=deepcopy(receipt);episode.update(target_id=row['target_id'],target=row['target_id'],
            windows=count,split_sample_range=[start,start+count],handoff_policy_id=rt.handoff.POLICY_ID)
        episode.update({k:deepcopy(row[k]) for k in ('pose_id','attempt_id','cohort') if k in row})
        episodes.append(episode)
        derived=deepcopy(sample['derived_index']);derived.update(episode_id=eid,split=split,
            split_sample_range=[start,start+count]);index['episodes'].append(derived)
        for j,window in enumerate(derived['windows']):
            index['samples'].append(dict(episode_id=eid,split=split,sample_index=start+j,
                episode_sample_index=j,window=deepcopy(window)))
        refs.append(dict(episode_id=eid,reference_sha256=meta['reference_sha256'],
            reference_revision=0,reference=deepcopy(meta['given_reference'])))
    need(all(parts.values()),'Positive TRAIN and selection-VAL required')
    result={}
    for split in parts:
        result[split]={k:np.concatenate([p[k] for p in parts[split]]) for k in ARRAY_KEYS}
        result[split].update(schema=given.MASKED_DATASET_SCHEMA,task_id=data.pose.TASK_ID,
            action_semantics_id=given.ACTION_SEMANTICS_ID,observation_contract=data.pose.observation_contract(),
            label_protocol=shared.MASKED_LABEL_PROTOCOL)
        given.require_samples(result[split])
    result.update(counts=offsets,episodes=episodes,derived_index=index,reference_inventory=refs,
        split_subject_ids={s:sorted({e['subject_id'] for e in episodes if e['split']==s}) for s in parts},
        normalization=given.fit_normalization(result['train'],split='train'))
    need(not set(result['split_subject_ids']['train'])&set(result['split_subject_ids']['validation']),
         'TRAIN/selection identities overlap')
    return result


def coverage(samples):
    state=samples['states'];mask=samples['action_is_pad'];distance=np.linalg.norm(state[:,7:10],axis=1)
    # Encoded rotation stores the first two columns of TCP^-1 @ goal.
    z=np.cross(state[:,10:13],state[:,13:16]);gap=(state[:,7:10]*z).sum(1)
    tangent=np.linalg.norm(state[:,7:10]-z*gap[:,None],axis=1)
    angle=np.rad2deg(np.arccos(np.clip(z[:,2],-1,1)))
    lengths=(~mask).sum(1)
    return dict(windows=len(state),valid_future_cells=int(lengths.sum()),
        future_length_histogram={str(n):int((lengths==n).sum()) for n in range(1,21)},
        goal_distance_minimum_m=float(distance.min()),tangent_minimum_m=float(tangent.min()),
        tangent_within_2mm=int((tangent<=.002).sum()),
        distance_bands={str(mm):dict(windows=int((distance<=mm/1000).sum()),
            axis_error_max_deg=(float(angle[distance<=mm/1000].max()) if np.any(distance<=mm/1000) else None))
            for mm in (10,20,35,60,100)},
        interpretation='Encoded uncompressed TCP vs prescribed reference, not measured contact-plane flatness. Origins inside the instantaneous handoff trigger are not required; policy is never queried there.')


def export(request_ref,output,*,cpu_workers=1):
    need(type(cpu_workers) is int and 1<=cpu_workers<=3,'One through three CPU decoder workers required')
    request=rt.read_bound(request_ref)
    need(set(request)=={'schema','source_dataset','recovery_manifests'} and request['schema']==REQUEST_SCHEMA,
         'Exact original source dataset plus explicit recovery manifests required')
    source=rt.read_bound(request['source_dataset'])
    need(source.get('schema')==trainer.PREPARATION_SCHEMA and source.get('complete') is True and
         source.get('rebuild_epoch')=='observed-reconstruction-20261105' and source.get('original_cases')==0 and
         source.get('task_contract')==rt.task_contract(),'Original qualified new-geometry rebuilt source required')
    nominal=rt.read_bound(source['source_manifest'])
    need(nominal.get('schema')==data.MANIFEST_SCHEMA,'Actual nominal source manifest required')
    rows=deepcopy(nominal['episodes']);manifests=request['recovery_manifests']
    need(isinstance(manifests,list) and len(manifests)<=64,'Explicit finite recovery manifest list required')
    for ref in manifests:
        manifest=rt.read_bound(ref)
        need(manifest.get('schema')==recovery.MANIFEST_SCHEMA and 1<=len(manifest.get('episodes',[]))<=4 and
             all(r.get('split')=='train' for r in manifest['episodes']),'TRAIN-only finite recovery manifests required')
        rows.extend(deepcopy(manifest['episodes']))
    base=Path.cwd();data._declarations(rows,base)
    data._reset_preflight(rows,base)
    for row in rows:
        rebuild.permitted_subject(row['split'],row['subject_id']);data._metadata(row,base)
    output=Path(output).resolve();need(not output.exists(),'New immutable masked export required');output.mkdir()
    rt.write_new(output/'REQUEST.json',request)
    execution=rt.write_new(output/'DECODE_EXECUTION.json',dict(cpu_workers=cpu_workers,
        decoder=rt.reference(__file__),input_order_preserved=True,original_physical_decoder=True))
    decoded=[];outcomes=[]
    try:
        for row,payload,error in decode_rows(rows,base,cpu_workers=cpu_workers):
            outcome=dict(episode_id=row['episode_id'],positive_data=False,windows=0)
            if payload is not None:
                decoded.append((row,payload))
                outcome.update(positive_data=True,windows=len(payload[0]['states']))
            else:outcome['error']=error
            outcomes.append(outcome)
            print(json.dumps(outcome),flush=True)
        result=combine(decoded);arrays={}
        for split in ('train','validation'):
            path=output/(split+'.npz')
            with path.open('xb') as stream:np.savez(stream,**{k:result[split][k] for k in ARRAY_KEYS})
            arrays[split]=rt.reference(path)
        ledger=rt.write_new(output/'outcomes.json',dict(original_source_outcomes=source['outcomes'],
            attempted_imports=outcomes,fixed_denominator=source['fixed_denominator']+len(rows)-len(nominal['episodes'])))
        positive=rt.write_new(output/'positive_manifest.json',dict(schema=data.MANIFEST_SCHEMA,
            episodes=[row for row,_ in decoded],label_protocol=shared.MASKED_LABEL_PROTOCOL))
        coverage_ref=rt.write_new(output/'COVERAGE.json',{s:coverage(result[s]) for s in ('train','validation')})
        result['derived_index']['manifest_sha256']=positive['sha256']
        complete=rt.write_new(output/'PREPARED.json',dict(schema=trainer.MASKED_PREPARATION_SCHEMA,complete=True,
            arrays=arrays,normalization=rt.write_new(output/'normalization.json',result['normalization']),
            index=rt.write_new(output/'index.json',result['derived_index']),episodes=result['episodes'],
            source_manifest=positive,source_dataset=request['source_dataset'],coverage=coverage_ref,outcomes=ledger,
            counts=result['counts'],split_subject_ids=result['split_subject_ids'],task_contract=rt.task_contract(),
            reference_inventory=result['reference_inventory'],rebuild_epoch='observed-reconstruction-20261105',
            decode_execution=execution,
            original_cases=0,**rebuild.FLAGS))
        trainer.load_dataset(complete)
        return rt.write_new(output/'COMPLETE.json',rt.read_bound(complete))
    except BaseException as error:
        (output/'traceback.txt').write_text(traceback.format_exc())
        rt.write_new(output/'FAILED.json',dict(error=repr(error),outcomes=outcomes,request=request_ref))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cpu-workers',type=int,default=1,choices=(1,2,3))
    args=p.parse_args();ref=rt.reference(args.request);need(ref['sha256']==args.sha256,'Request changed')
    export(ref,args.output,cpu_workers=args.cpu_workers)


if __name__=='__main__':main()
