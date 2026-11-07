"""Positive-only diagnostic recovery labels; nominal formal admission unchanged.

  python -m sim.ausc_policy.m2_given_contact_recovery_dataset \
    --manifest FILE --sha256 SHA --output NEW_DIRECTORY

Manifest schema m2_given_contact_recovery_manifest_v1 has the usual explicit
given dataset episode declarations. Only TRAIN may appear. Each run binds its
own given_recovery.json through result.recovery_collection. Owners retain the
original maximum four children. No simulator, model, TEST, or formal permit.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from . import m2_given_contact_dataset as data, m2_given_contact_act as act
from . import m2_given_contact_recovery as recovery, m2_given_contact_runtime as runtime

need=data.need
MANIFEST_SCHEMA='m2_given_contact_recovery_manifest_v1'
EXPORT_SCHEMA='m2_given_contact_recovery_preparation_v1'


def validate_sidecar(sidecar,audit,metadata):
    """Bind every actual setter interval and role; unknown is never teacher."""
    need(sidecar.get('schema')==recovery.SIDECAR_SCHEMA and sidecar.get('completed') is True and
         sidecar.get('episode_id')==metadata['episode_id'] and
         sidecar.get('reference_sha256')==metadata['reference_sha256'] and
         type(sidecar.get('native_steps')) is int and sidecar.get('native_steps')==len(audit) and
         type(sidecar.get('perturbations_executed')) is int and sidecar.get('perturbations_executed')==1,
         'Complete actual recovery sidecar/episode/reference required')
    entries=sidecar.get('events');need(isinstance(entries,list),'Recovery setter role ledger required')
    setters=[];ends={}
    for event in audit:
        raw=event.get('raw_intent')
        if raw is not None:
            if raw['native_tick_in_cell']==0:setters.append(raw['setter'])
            ends[raw['setter']['intent_id']]=event['physics_row']+1
    need(len(entries)==len(setters) and entries,'Missing recovery setter role provenance')
    by_row={};actual_setters={};student=sidecar.get('declaration',{}).get('schema')==recovery.STUDENT_SCHEMA
    for i,(entry,setter) in enumerate(zip(entries,setters)):
        need(set(entry)=={'intent_id','setter_sha256','physics_row_start','physics_row_end_exclusive','kind','completed'} and
             entry['kind'] in recovery.KINDS and type(entry['completed']) is bool and
             all(type(entry[k]) is int for k in ('intent_id','physics_row_start','physics_row_end_exclusive')),
             'Unknown recovery setter role/fields')
        start=setter['physics_row_start'];end=ends[setter['intent_id']]
        need(entry['intent_id']==setter['intent_id'] and entry['setter_sha256']==setter['setter_sha256'] and
             entry['physics_row_start']==start and entry['physics_row_end_exclusive']==end and end>start and
             entry['completed']==(end-start==12), 'Recovery role ledger differs from actual setter hash/interval')
        if entry['kind']!='excluded':
            student_wait=student and entry['kind']=='perturbation' and setter['execution_role']=='given_l2_history_wait'
            need(not setter['offgrid_stop'] and setter['control_owner']=='TEACHER_L2' and
                 (setter['execution_role']=='given_l2_motion' or student_wait), 'STOP/ALIGN/nonteacher cannot provide recovery labels')
        by_row[start]=entry;actual_setters[start]=setter
    perturbed=[e for e in entries if e['kind']=='perturbation']
    expected=recovery.perturbation_cell_count(sidecar.get('declaration',{}))
    need(len(perturbed)==expected and all(e['completed'] for e in perturbed) and
         all(b['physics_row_start']==a['physics_row_end_exclusive'] for a,b in zip(perturbed,perturbed[1:])),
         'One complete actual declared perturbation interval required')
    if student:
        prefix=runtime.read_bound(sidecar['declaration']['student_prefix'])
        need(len(prefix.get('source_roles',[]))==expected,'Retained source role count differs')
        for entry,target,role in zip(perturbed,prefix['raw_targets'],prefix['source_roles']):
            setter=actual_setters[entry['physics_row_start']]
            need(role in ('given_l2_motion','given_l2_history_wait') and setter['execution_role']==role and
                np.array_equal(setter['target_q'],target),'Actual student replay differs from its retained target/role input')
    return by_row


def filter_recovery_samples(samples,audit,metadata,sidecar):
    by_row=validate_sidecar(sidecar,audit,metadata);keep=[]
    for i,window in enumerate(samples['derived_index']['windows']):
        first=window['inclusive_native_ticks'][0]+1
        entries=[by_row.get(first+k*12) for k in range(window.get('valid_future_cells',20))]
        if any(e is None or not e['completed'] or e['kind'] not in ('nominal_teacher','recovery_teacher') for e in entries):continue
        if not any(e['kind']=='recovery_teacher' for e in entries):continue
        keep.append(i)
    result=deepcopy(samples)
    keys=runtime.shared.ARRAY_KEYS+(('action_is_pad',) if 'action_is_pad' in samples else ())
    for key in keys:result[key]=samples[key][keep].copy()
    result['derived_index']['windows']=[deepcopy(samples['derived_index']['windows'][i]) for i in keep]
    result.update(recovery_supervision=True,training_eligible=False,physical_qualification=False)
    for window in result['derived_index']['windows']:
        window['label_role']='actual_clean_recovery_teacher_raw_intents'
    return result


def assemble_recovery_samples(records,audit,metadata,sidecar):
    need(metadata.get('split')=='train','Recovery labels are TRAIN only')
    return filter_recovery_samples(act.assemble_samples(records,audit,metadata),audit,metadata,sidecar)


def require_sidecar_location(ref,output):
    need(Path(ref['path'])==Path(output)/'fresh_expert/given_recovery.json',
         'Recovery sidecar must be the actual native fresh_expert output')


def read_episode(row,base, *, allow_truncated=False):
    """New explicit class admission; original full physical decoder stays intact."""
    need(row.get('split')=='train','Recovery export is TRAIN only')
    data._declarations([row],base);context=data._metadata(row,base)
    run,config,freeze=data._native_evidence(row,base,context[2],physical=False)
    binding=recovery.preflight(config)
    child,_=data._bound(row['child_result'],base)
    need(child.get('physical_pass') is True and run.get('success') is True and run.get('physical_validation') is True and
         run.get('errors')==[] and run.get('controller_motion_complete') is True and run.get('compact_capture_complete') is True and
         run.get('executed_controller_class')==recovery.__name__+'.GivenContactRecoveryExpert',
         'Complete physically passing current recovery teacher required')
    ref=run.get('recovery_collection');sidecar=runtime.read_bound(ref)
    output=data._ref_path(row['run_result'],base).parent
    require_sidecar_location(ref,output)
    need(sidecar.get('config_sha256')==runtime.digest(config) and
         sidecar.get('declaration')==binding['settings'], 'Recovery sidecar/config binding differs')
    # This invokes the original per-tick physical, parking, ALIGN, HOLD, release,
    # source-freeze, full capture fidelity and final-home checks without edits.
    samples,receipt,metadata=data._decode_capture(row,base,context,run,config,freeze,allow_truncated=allow_truncated)
    with gzip.open(context[0]/'native_audit.jsonl.gz','rt') as stream:audit=[json.loads(line) for line in stream]
    result=filter_recovery_samples(samples,audit,metadata,sidecar)
    need(len(result['states'])>0,'No complete clean teacher recovery prefix windows')
    receipt.update(windows=len(result['states']),recovery_collection=ref)
    return result,receipt,metadata


def export_dataset(manifest_ref,output):
    manifest=runtime.read_bound(manifest_ref);rows=manifest.get('episodes')
    need(manifest.get('schema')==MANIFEST_SCHEMA and isinstance(rows,list) and 1<=len(rows)<=4,
         'Explicit finite one-to-four-case recovery pilot manifest required')
    need(all(r.get('split')=='train' for r in rows),'Recovery export is TRAIN only; no VAL/TEST')
    base=Path(manifest_ref['path']).parent;data._declarations(rows,base)
    output=Path(output);need(not output.exists(),'New recovery export directory required');output.mkdir(parents=True)
    arrays=[];outcomes=[];windows=[];offset=0;resets=set()
    for row in rows:
        outcome=dict(episode_id=row['episode_id'],positive_data=False,windows=0)
        try:
            sample,receipt,metadata=read_episode(row,base)
            identity=receipt.get('native_reset_identity',{})
            need(type(identity.get('pid')) is int and type(identity.get('sid')) is int and
                 isinstance(identity.get('start_ticks'),str),'Native reset process identity missing')
            reset=(identity['pid'],identity['sid'],identity['start_ticks'])
            need(reset not in resets,'Duplicate native process reused across episode resets')
            resets.add(reset)
            arrays.append(sample)
            windows.extend(dict(episode_id=row['episode_id'],sample_index=offset+i,window=w)
                for i,w in enumerate(sample['derived_index']['windows']))
            offset+=len(sample['states']);outcome.update(positive_data=True,windows=len(sample['states']),receipt=receipt)
        except Exception as error:outcome['error']=type(error).__name__+': '+str(error)
        outcomes.append(outcome)
    refs={}
    if arrays:
        train={key:np.concatenate([a[key] for a in arrays]) for key in runtime.shared.ARRAY_KEYS}
        with (output/'train.npz').open('xb') as stream:np.savez(stream,**train)
        train.update(schema=act.DATASET_SCHEMA,task_id=runtime.pose.TASK_ID,
            action_semantics_id=act.ACTION_SEMANTICS_ID,observation_contract=runtime.pose.observation_contract())
        refs.update(train=runtime.reference(output/'train.npz'),normalization=runtime.write_new(output/'normalization.json',
            act.fit_normalization(train,split='train')))
    refs['index']=runtime.write_new(output/'index.json',dict(samples=windows,source_manifest=manifest_ref))
    return runtime.write_new(output/'COMPLETE.json',dict(schema=EXPORT_SCHEMA,completed=True,source_manifest=manifest_ref,
        source_module=runtime.reference(__file__),outputs=refs,attempted_cases=len(rows),positive_cases=len(arrays),
        windows=offset,outcomes=outcomes,diagnostic_only=True,training_eligible=False,formal_acceptance=False,
        test_payloads_opened=False,model_qualified=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    ref=runtime.reference(args.manifest);need(ref['sha256']==args.sha256,'Recovery manifest digest differs')
    export_dataset(ref,args.output)


if __name__=='__main__':main()
