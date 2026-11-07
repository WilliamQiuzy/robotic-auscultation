"""Bounded diagnostic expert capture and fresh ACT arrays after the Pod loss.

This helper only prepares an existing guard-owned isaac_batch, or performs
CPU postflight after that owner is terminal. It never launches native physics,
modifies a gate, reads original-eight arrays, or opens frozen-evaluation scenes.
All requested subject/pose/target slots survive in the attempt ledger. Only
the unchanged given-contact native/physical/fidelity/window reader can produce
positive data. A complete preparation remains diagnostic and formally ineligible.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

import numpy as np

from . import m2_given_contact_dataset as data
from . import m2_given_contact_runtime as runtime
from . import m2_pilot_config as pilot

need=runtime.need
REQUEST_SCHEMA='m2_rebuild_expert_capture_request_v1'
BUILD_SCHEMA='m2_rebuild_expert_capture_preparation_v1'
LEDGER_SCHEMA='m2_rebuild_expert_capture_outcomes_v1'
PREPARATION_SCHEMA='m2_given_diverse_pilot_preparation_v1'
TRAIN_IDS=('p001120','p000758','p001031','p000833')
SELECTION_ID='p000923'
FROZEN_IDS=('p001030','p000956')
POSE_IDS=('p000474_a001230_f0000','p000476_a001231_f0023')
FLAGS=dict(diagnostic_only=True,formal_acceptance=False,training_eligible=False,
    formal_population_qualified=False,test_payloads_opened=False,model_qualified=False,
    real_hardware_qualified=False,physics_gates_changed=False)


def permitted_subject(split,subject):
    need(subject not in FROZEN_IDS,'Frozen evaluation identity is excluded from capture/export/selection')
    need((split=='train' and subject in TRAIN_IDS) or (split=='validation' and subject==SELECTION_ID),
         'Only the explicit rebuilt TRAIN/selection-validation identities are permitted')
    data.shared.require_membership(split,subject)


def pilot_ref(ref):
    """Translate only the reference spelling; authenticate the original bytes."""
    path=runtime.bound_path(ref)
    return dict(file=str(path),sha256=ref['sha256'])


def validate_request(value):
    keys={'schema','run_id','base','source','reconstruction_source','subject_ids','targets','scenes',
          'parallelism','time_limit_seconds','max_frames','retainer_witness_contract','scene_store'}
    need(isinstance(value,dict) and set(value)==keys and value.get('schema')==REQUEST_SCHEMA,
         'Explicit rebuild request required; gate/control overrides are forbidden')
    run=value['run_id']
    need(isinstance(run,str) and re.fullmatch('[a-z0-9][a-z0-9_-]{0,39}',run) is not None,
         'Bounded original guard-compatible run ID required')
    need(isinstance(value['base'],str) and Path(value['base']).is_absolute(),'Absolute owner base required')
    subjects=value['subject_ids'];targets=value['targets']
    need(isinstance(subjects,list) and len(subjects)==len(set(subjects)) and SELECTION_ID in subjects and
         2<=len(subjects)<=5 and set(subjects)<=(set(TRAIN_IDS)|{SELECTION_ID}),
         'Explicit nonempty TRAIN subset plus selection-validation ID required; frozen IDs are excluded')
    need(isinstance(targets,list) and 2<=len(targets)<=4 and len(targets)==len(set(targets)) and
         all(t in {s+str(i) for s in 'LR' for i in range(1,6)} for t in targets),
         'Explicit two-to-four unique targets required; no silent target selection')
    for key,low,high in [('parallelism',1,4),('time_limit_seconds',121,86400),('max_frames',1,6000)]:
        need(type(value[key]) is int and low<=value[key]<=high,'Finite bounded '+key+' required')
    need(isinstance(value['retainer_witness_contract'],dict) and value['retainer_witness_contract'],
         'Original qualified native retainer witness contract is mandatory')
    store=value['scene_store']
    need(isinstance(store,dict) and set(store) in ({'root','batch_id'},{'root','batch_id','sdk_mdl_manifest'}) and
         isinstance(store['root'],str) and Path(store['root']).is_absolute() and store['batch_id']==run,
         'Original shared scene-store binding for this run is mandatory')
    scenes=value['scenes'];seen=set();rows=[]
    need(isinstance(scenes,list) and len(scenes)==len(subjects)*len(POSE_IDS),'Every declared subject needs both observed poses')
    for scene in scenes:
        scene_keys={'subject_id','split','cohort','pose_id','preparation_config','prepared_complete','patient_receipt','plans'}
        need(isinstance(scene,dict) and set(scene) in (scene_keys,scene_keys|{'preparation_error'}),
             'Explicit prepared scene fields required')
        if 'preparation_error' in scene:
            need(isinstance(scene['preparation_error'],str) and bool(scene['preparation_error'].strip()) and
                 all(isinstance(plan,dict) and plan.get('status')=='unavailable' for plan in scene['plans'].values()),
                 'Failed scene preparation must retain its reason and mark every target unavailable')
        subject=scene['subject_id'];split=scene['split'];pose=scene['pose_id']
        permitted_subject(split,subject)
        need(subject in subjects and pose in POSE_IDS and (subject,pose) not in seen,
             'Unexpected/duplicate subject-pose scene; denominator cannot be reassigned')
        seen.add((subject,pose))
        need(scene['cohort']==('train' if split=='train' else 'selection_validation'),
             'Exact optimizer/selection role required; frozen evaluation scenes are excluded')
        need(isinstance(scene['plans'],dict) and set(scene['plans'])==set(targets),
             'Every explicit target needs a plan or a retained unavailable reason')
        for target in targets:
            plan=scene['plans'][target]
            need(isinstance(plan,dict) and (set(plan)=={'plan'} or
                 set(plan)=={'status','reason'} and plan['status']=='unavailable' and
                 isinstance(plan['reason'],str) and bool(plan['reason'].strip())),
                 'Hash-bound plan or explicit unavailable reason required')
            job=subject+'-'+pose+'-'+target.lower()
            rows.append(dict(attempt_id=job,job_id=job,episode_id=run+'-'+job,
                reset_id=run+'-'+job+'-reset',subject_id=subject,split=split,
                cohort=scene['cohort'],pose_id=pose,target_id=target,target=target))
    need(seen=={(subject,pose) for subject in subjects for pose in POSE_IDS},'Declared scene Cartesian scope differs')
    return rows


def validate_sources(value):
    source=value['source']
    need(isinstance(source,dict) and set(source)=={'code_commit','code_commit_file','source_manifest'} and
         isinstance(source['code_commit'],str) and re.fullmatch('[0-9a-f]{40}',source['code_commit']) is not None,
         'Explicit immutable collector source archive required')
    commit=runtime.bound_path(source['code_commit_file']);manifest=runtime.bound_path(source['source_manifest'])
    need(commit.name=='CODE_COMMIT' and manifest.name=='SOURCE_MANIFEST.json' and commit.parent==manifest.parent and
         commit.read_text().strip()==source['code_commit'] and Path(pilot.__file__).resolve().parents[2]==commit.parent,
         'Build must import the exact archived collector code root')
    rows=runtime.read_bound(source['source_manifest']);need(isinstance(rows,list) and rows,'Source inventory required')
    names=set()
    for row in rows:
        name=row.get('file')
        need(isinstance(name,str) and not Path(name).is_absolute() and name not in names,'Unique archived source path required')
        names.add(name);path=data._path(name,commit.parent)
        need(commit.parent in path.parents and path.is_file() and data.sha256(path)==row.get('sha256') and
             ('bytes' not in row or path.stat().st_size==row['bytes']),'Archived collector source changed: '+name)
    reconstruction=runtime.read_bound(value['reconstruction_source'])
    need(reconstruction.get('schema')=='m2_observed_reconstruction_sources_v1',
         'New observed reconstruction source required; old depth-plane bytes were lost')


def scene_metadata(scene,value):
    """Check every patient/config header before plan or scene payload assembly."""
    permitted_subject(scene['split'],scene['subject_id'])
    prep=runtime.read_bound(scene['preparation_config']);complete=runtime.read_bound(scene['prepared_complete'])
    patient=runtime.read_bound(scene['patient_receipt']);spec=prep.get('patient_asset',{})
    expected=pilot_ref(value['reconstruction_source'])
    need(patient.get('identity')==scene['subject_id'] and patient.get('split')==scene['split'] and
         complete.get('patient')==patient and spec.get('expected_split')==scene['split'],
         'Prepared patient/config identity differs from explicit rebuilt scope')
    need(spec.get('reconstruction_source')==patient.get('reconstruction_source')==expected and
         spec.get('reconstruction_cohort')==patient.get('reconstruction_cohort')==scene['cohort'] and
         patient.get('original_depth_plane_bytes_recovered') is False,
         'Explicit new reconstruction source/cohort receipt required')
    need(Path(str(spec.get('file',''))).stem==scene['subject_id']+'_'+scene['pose_id'],
         'Prepared asset observed pose differs from requested subject/pose slot')


def build_batch(value,output):
    rows=validate_request(value);validate_sources(value)
    for scene in value['scenes']:
        if 'preparation_error' not in scene:scene_metadata(scene,value)
    output=Path(output).resolve();need(not output.exists(),'Fresh immutable build directory required')
    output.mkdir(parents=True);request=runtime.write_new(output/'REQUEST.json',value)
    attempts=[];jobs=[]
    for index,case in enumerate(rows):
        scene=value['scenes'][index//len(value['targets'])];plan_spec=scene['plans'][case['target_id']]
        attempt=dict(case,preparation_status='unavailable',reason=plan_spec.get('reason'),config=None)
        attempts.append(attempt)
        if 'preparation_error' in scene:
            attempt.update(preparation_status='scene_preparation_failed',reason=scene['preparation_error']);continue
        if 'plan' not in plan_spec:continue
        attempt['fresh_plan']=deepcopy(plan_spec['plan'])
        try:
            plan=runtime.read_bound(plan_spec['plan'])
            need(plan.get('name')==case['target_id'],'Fresh plan target differs from explicit request')
            if plan.get('success') is not True:
                attempt.update(preparation_status='planning_failed',reason='Hash-bound fresh plan did not succeed');continue
            assembled=pilot.build_given_contact_expert(
                **{key:pilot_ref(scene[key]) for key in ('preparation_config','prepared_complete','patient_receipt')},
                fresh_plan=pilot_ref(plan_spec['plan']),episode_id=case['episode_id'],max_frames=value['max_frames'],
                collision_profile='full',retainer_witness_contract=deepcopy(value['retainer_witness_contract']),
                scene_store=deepcopy(value['scene_store']))
            config=assembled['config']
            need(config.get('base')==value['base'] and type(config.get('time_limit_seconds')) is int and
                 60<config['time_limit_seconds']<=value['time_limit_seconds'],'Original bounded child/base contract differs')
            ref=runtime.write_new(output/'configs'/(case['job_id']+'.json'),config)
            receipt=runtime.write_new(output/'configs'/(case['job_id']+'.receipt.json'),assembled['receipt'])
            jobs.append(dict(id=case['job_id'],config=ref['path'],sha256=ref['sha256']))
            attempt.update(preparation_status='runnable',reason=None,config=ref,config_receipt=receipt)
        except Exception as error:
            attempt.update(preparation_status='configuration_failed',reason=type(error).__name__+': '+str(error))
    batch=None
    if jobs:
        batch=runtime.write_new(output/'batch.json',dict(experiment='isaac_batch',base=value['base'],
            parallelism=value['parallelism'],time_limit_seconds=value['time_limit_seconds'],jobs=jobs))
    comparison=runtime.write_new(output/'COMPARISON_SCOPE.json',dict(schema='m2_rebuild_comparison_scope_v1',
        request=request,arms=['expert','act','astra'],frozen_before_evaluation=True,
        metadata_is_prelaunch_intent_not_execution_evidence=True,fixed_denominator=len(rows),cases=deepcopy(attempts),
        target_ids=deepcopy(value['targets']),frozen_evaluation_subject_ids_excluded=list(FROZEN_IDS),
        evaluation_roles={'train':'in_sample_diagnostic','selection_validation':'selection_diagnostic'},
        pairing='same prepared scene, observed pose, prescribed target and original physical config',**FLAGS))
    validate_sources(value)
    result=dict(schema=BUILD_SCHEMA,complete=True,request=request,run_id=value['run_id'],source=value['source'],batch=batch,
        comparison_scope=comparison,fixed_denominator=len(rows),runnable_cases=len(jobs),attempts=attempts,physical_validation=False,
        task_contract=runtime.task_contract(),**FLAGS)
    runtime.write_new(output/'COMPLETE.json',result)
    return result


def source_freeze(report,capture,source,output,job):
    sources={}
    for mapping in (report.get('runtime_sources_sha256'),capture.get('runtime_sources_sha256')):
        need(isinstance(mapping,dict) and mapping,'Actual run/capture native source inventories required')
        for path,digest in mapping.items():
            need(path not in sources or sources[path]==digest,'Conflicting actual native source digest');sources[path]=digest
    files={}
    for path,digest in sorted(sources.items()):
        ref=runtime.reference(path);need(ref['sha256']==digest,'Actual native source changed: '+path);files[path]=ref
    value=dict(schema=data.SOURCE_FREEZE_SCHEMA,files=files,**deepcopy(source))
    ref=runtime.write_new(output/(job+'-source-freeze.json'),value)
    data._source_freeze(ref,Path.cwd(),sources)
    return ref


def case_headers(prepared,value,owner_ref,output):
    owner=runtime.read_bound(owner_ref);batch=runtime.read_bound(prepared['batch']) if prepared['batch'] else None
    need(batch is not None and owner.get('config')==batch and owner.get('run_id')==value['run_id'] and
         owner.get('code_commit')==value['source']['code_commit'] and isinstance(owner.get('ended_utc'),str) and
         owner.get('gpu_released') is True and owner.get('stage',{}).get('process_group_drained') is True,
         'Exact terminal drained/released original owner/config/source required before CPU postflight')
    run=Path(owner_ref['path']).parent;entries=[]
    for attempt in prepared['attempts']:
        entry=dict(case=deepcopy(attempt),header_error=None,reported_status={});entries.append(entry)
        if attempt['preparation_status']!='runnable':entry['header_error']=attempt['reason'];continue
        case=attempt;root=run/'output'/case['job_id']
        status=entry['reported_status']
        for name,path in [('launch',root.parent/(case['job_id']+'.launch.json')),
                          ('child_result',root.parent/(case['job_id']+'.result.json')),
                          ('partial_result',root/'fresh_expert_result.json'),('run_result',root/'COMPLETE.json')]:
            if path.is_file():status[name]=runtime.reference(path)
        status['launched']='launch' in status
        try:
            child=runtime.read_bound(status['child_result']);report=runtime.read_bound(status['run_result'])
            status.update(worker_complete=child.get('worker_complete'),physical_pass=child.get('physical_pass'),
                success=report.get('success'),physical_validation=report.get('physical_validation'),errors=report.get('errors'))
            if 'partial_result' in status:
                fresh=runtime.read_bound(status['partial_result'])
                need(data.pose.canonical({key:report[key] for key in fresh})==data.pose.canonical(fresh),
                     'Native COMPLETE differs from original fresh-expert result')
            config=runtime.read_bound(case['config']);meta=config.get('m2_given_contact_control',{}).get('metadata',{})
            permitted_subject(meta.get('split'),meta.get('subject_id'))
            need(all(meta.get(key)==case[key] for key in ('episode_id','subject_id','split','target_id')),
                 'Actual collector config identity differs from requested attempt')
            capture=root/'compact_multimodal';header=data.shared._json(capture/'manifest.json')
            compact=runtime.reference(capture/'compact/manifest.json')
            freeze=source_freeze(report,header,value['source'],output,case['job_id'])
            row={key:deepcopy(case[key]) for key in ('episode_id','subject_id','split','target_id','target',
                'reset_id','job_id','config','attempt_id','cohort','pose_id')}
            row.update(run_id=value['run_id'],owner_terminal=owner_ref,child_result=status['child_result'],
                run_result=status['run_result'],capture_root=str(capture),source_freeze=freeze,
                artifacts={'compact/manifest.json':{key:compact[key] for key in ('bytes','sha256')}})
            entry.update(row=row,context=data._metadata(row,Path.cwd()))
        except Exception as error:entry['header_error']=type(error).__name__+': '+str(error)
    admitted=[e['row'] for e in entries if e['header_error'] is None]
    if admitted:
        try:data._declarations(admitted,Path.cwd());data._reset_preflight(admitted,Path.cwd())
        except Exception as error:
            for entry in entries:
                if entry['header_error'] is None:entry['header_error']='Cross-episode preflight: '+str(error)
    return entries


def postflight(entries,base):
    decoded=[];outcomes=[]
    for order,entry in enumerate(entries):
        case=entry['case'];outcome=deepcopy(case)
        outcome.update(selection_row=order,positive_data=False,windows=0,error=entry.get('header_error'),
            reported_status=deepcopy(entry.get('reported_status',{})))
        outcomes.append(outcome)
        if outcome['error'] is not None:continue
        try:
            row=entry['row'];permitted_subject(row['split'],row['subject_id'])
            if 'capture_root' in row:
                root=data._path(row['capture_root'],base);inventory={}
                for path in sorted(root.rglob('*')):
                    need(not path.is_symlink(),'Symlink in capture inventory')
                    if path.is_file():
                        ref=runtime.reference(path);inventory[path.relative_to(root).as_posix()]={k:ref[k] for k in ('bytes','sha256')}
                row['artifacts']=inventory
            # This exact original reader checks owner, physics, witness, camera,
            # source, storage, fidelity, causal roles and positive phase windows.
            value=data._read_payload(row,base,entry['context']);data.given.require_samples(value[0])
            count=len(value[0]['states']);need(count>0,'No actual complete H20 positive windows')
            decoded.append((row,value));outcome.update(positive_data=True,windows=count)
        except Exception as error:outcome['error']=type(error).__name__+': '+str(error)
    return decoded,outcomes


def combine(decoded):
    parts={split:[] for split in ('train','validation')};offsets={split:0 for split in parts}
    episodes=[];index=dict(schema=data.INDEX_SCHEMA,task_id=data.pose.TASK_ID,
        action_semantics_id=data.given.ACTION_SEMANTICS_ID,episodes=[],samples=[])
    references=[];seen=set()
    for row,(samples,receipt,meta) in decoded:
        permitted_subject(row['split'],row['subject_id']);data.given.require_samples(samples)
        eid=row['episode_id'];need(eid not in seen,'Duplicate fresh episode identity');seen.add(eid)
        split=row['split'];count=len(samples['states']);need(count>0,'Positive H20 windows required')
        start=offsets[split];offsets[split]+=count;parts[split].append(samples);order=len(episodes)
        derived=deepcopy(samples['derived_index'])
        need(len(derived.get('windows',[]))==count,'Complete original causal window index required')
        derived.update(episode_id=eid,manifest_row=order,split=split,split_sample_range=[start,start+count],
            reference_sha256=meta['reference_sha256']);index['episodes'].append(derived)
        receipt=deepcopy(receipt);receipt.update(target_id=row['target_id'],target=row['target_id'],
            origin='rebuilt_actual_native_original_physical_reader',split_sample_range=[start,start+count],
            handoff_policy_id=data.handoff.POLICY_ID);episodes.append(receipt)
        receipt.update({key:deepcopy(row[key]) for key in ('pose_id','attempt_id','cohort') if key in row})
        if 'pose_id' in row:receipt['observed_pose_id']=row['pose_id']
        if 'attempt_id' in row:receipt['source_attempt_id']=row['attempt_id']
        for local,window in enumerate(derived['windows']):
            index['samples'].append(dict(episode_id=eid,manifest_row=order,split=split,sample_index=start+local,
                episode_sample_index=local,reference_sha256=meta['reference_sha256'],window=deepcopy(window)))
        references.append(dict(episode_id=eid,reference_sha256=meta['reference_sha256'],reference_revision=0,
            reference=deepcopy(meta['given_reference'])))
    need(all(parts.values()),'At least one actual positive TRAIN and selection-validation episode required')
    result={}
    for split in parts:
        result[split]={key:np.concatenate([part[key] for part in parts[split]],axis=0) for key in data.shared.ARRAY_KEYS}
        result[split].update(schema=data.given.DATASET_SCHEMA,task_id=data.pose.TASK_ID,
            action_semantics_id=data.given.ACTION_SEMANTICS_ID,observation_contract=data.pose.observation_contract())
    result.update(counts=offsets,episodes=episodes,derived_index=index,reference_inventory=references,
        split_subject_ids={split:sorted({e['subject_id'] for e in episodes if e['split']==split}) for split in parts},
        normalization=data.given.fit_normalization(result['train'],split='train'))
    return result


def publish_arrays(output,result,prepared,outcomes,provenance):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arrays={};digests={}
    for split in ('train','validation'):
        path=output/(split+'.npz')
        with path.open('xb') as stream:np.savez(stream,**{key:result[split][key] for key in data.shared.ARRAY_KEYS})
        arrays[split]=runtime.reference(path)
        digests[split]={key:dict(shape=list(result[split][key].shape),dtype=result[split][key].dtype.str,
            sha256=hashlib.sha256(np.ascontiguousarray(result[split][key]).tobytes()).hexdigest()) for key in data.shared.ARRAY_KEYS}
    manifest=runtime.write_new(output/'positive_manifest.json',dict(schema=data.MANIFEST_SCHEMA,
        task_id=data.pose.TASK_ID,action_semantics_id=data.given.ACTION_SEMANTICS_ID,
        handoff_policy_id=data.handoff.POLICY_ID,episodes=provenance.get('positive_source_rows',[]),**FLAGS))
    result['derived_index']['manifest_sha256']=manifest['sha256']
    return runtime.write_new(output/'COMPLETE.json',dict(schema=PREPARATION_SCHEMA,complete=True,
        rebuild_epoch='observed-reconstruction-20261105',source_manifest=manifest,arrays=arrays,array_digests=digests,
        normalization=runtime.write_new(output/'normalization.json',result['normalization']),
        index=runtime.write_new(output/'index.json',result['derived_index']),task_contract=runtime.task_contract(),
        episodes=result['episodes'],reference_inventory=result['reference_inventory'],counts=result['counts'],
        split_subject_ids=result['split_subject_ids'],fixed_denominator=prepared['fixed_denominator'],
        positive_cases=len(result['episodes']),rejected_cases=prepared['fixed_denominator']-len(result['episodes']),
        original_cases=0,outcomes=outcomes,provenance=provenance,**FLAGS))


def training_config(base,dataset,*,steps=5000,eval_every=500,seed=20261105):
    from . import m2_given_contact_expanded_pilot_train as trainer
    value=dict(trainer.SCHEDULE,base=str(base),dataset=deepcopy(dataset),steps=steps,eval_every=eval_every,seed=seed)
    return trainer.validate_config(value)


def export(prepared_ref,owner_ref,output,*,steps=5000,eval_every=500,seed=20261105):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only CUDA_VISIBLE_DEVICES required')
    prepared=runtime.read_bound(prepared_ref);value=runtime.read_bound(prepared['request']);rows=validate_request(value)
    need(prepared.get('schema')==BUILD_SCHEMA and prepared.get('complete') is True and
         prepared.get('fixed_denominator')==len(rows) and len(prepared.get('attempts',[]))==len(rows) and
         prepared.get('task_contract')==runtime.task_contract() and prepared.get('source')==value['source'] and
         all(prepared.get(key) is wanted for key,wanted in FLAGS.items()),
         'Exact completed rebuild batch preparation/denominator required')
    for case,attempt in zip(rows,prepared['attempts']):
        need(all(attempt.get(key)==item for key,item in case.items()),'Ordered prelaunch attempt identity differs')
    comparison=runtime.read_bound(prepared['comparison_scope'])
    need(comparison.get('request')==prepared['request'] and comparison.get('cases')==prepared['attempts'] and
         comparison.get('fixed_denominator')==len(rows),'Immutable paired-comparison scope differs')
    validate_sources(value)
    for scene in value['scenes']:
        if 'preparation_error' not in scene:scene_metadata(scene,value)
    output=Path(output).resolve();need(not output.exists(),'Fresh immutable postflight directory required')
    output.mkdir(parents=True)
    request=runtime.write_new(output/'REQUEST.json',dict(preparation=prepared_ref,owner_terminal=owner_ref,
        helper=runtime.reference(Path(__file__)),task_contract=runtime.task_contract(),**FLAGS))
    try:
        entries=case_headers(prepared,value,owner_ref,output/'source_freezes')
        decoded,outcomes=postflight(entries,Path.cwd())
        ledger=runtime.write_new(output/'outcomes.json',dict(schema=LEDGER_SCHEMA,preparation=prepared_ref,
            owner_terminal=owner_ref,fixed_denominator=len(rows),outcomes=outcomes,**FLAGS))
        result=combine(decoded)
        validate_sources(value);runtime.read_bound(owner_ref)
        ref=publish_arrays(output/'dataset',result,prepared,outcomes,dict(request=request,attempt_ledger=ledger,
            preparation=prepared_ref,owner_terminal=owner_ref,positive_source_rows=[row for row,_ in decoded],
            comparison_scope=prepared['comparison_scope'],
            original_physical_reader='sim.ausc_policy.m2_given_contact_dataset._read_payload',
            original_label_assembler='sim.ausc_policy.m2_given_contact_act.assemble_samples',
            original_fidelity_reader='sim.ausc_policy.training_capture.iter_training_records'))
        from . import m2_given_contact_expanded_pilot_train as trainer
        trainer.load_dataset(ref)
        config=runtime.write_new(output/'act_config.json',training_config(value['base'],ref,steps=steps,eval_every=eval_every,seed=seed))
        return runtime.write_new(output/'COMPLETE.json',dict(schema='m2_rebuild_act_export_v1',complete=True,
            dataset=ref,act_config=config,attempt_ledger=ledger,counts=result['counts'],request=request,**FLAGS))
    except BaseException as error:
        (output/'traceback.txt').write_text(traceback.format_exc())
        runtime.write_new(output/'FAILED.json',dict(complete=False,request=request,error=type(error).__name__+': '+str(error),**FLAGS))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build');build.add_argument('--request',required=True);build.add_argument('--sha256',required=True)
    build.add_argument('--output',required=True)
    post=sub.add_parser('export');post.add_argument('--preparation',required=True);post.add_argument('--sha256',required=True)
    post.add_argument('--owner',required=True);post.add_argument('--owner-sha256',required=True);post.add_argument('--output',required=True)
    for key,default in [('steps',5000),('eval-every',500),('seed',20261105)]:post.add_argument('--'+key,type=int,default=default)
    args=parser.parse_args()
    if args.command=='build':
        value=runtime.read_bound(dict(path=args.request,sha256=args.sha256));result=build_batch(value,args.output)
        print(json.dumps(dict(output=str(Path(args.output).resolve()/'COMPLETE.json'),
            fixed_denominator=result['fixed_denominator'],runnable_cases=result['runnable_cases'])))
    else:
        print(json.dumps(export(dict(path=args.preparation,sha256=args.sha256),
            dict(path=args.owner,sha256=args.owner_sha256),args.output,steps=args.steps,eval_every=args.eval_every,seed=args.seed)))


if __name__=='__main__':main()
