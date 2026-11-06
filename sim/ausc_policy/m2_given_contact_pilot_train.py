"""Official ACT eight-case learning pilot; diagnostic TRAIN/VAL only, no resume.

This route consumes the already reconstructed given-contact arrays. It does
not qualify the formal population or produce a formal deployment checkpoint.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import numpy as np

from . import m2_given_contact_runtime as runtime

need=runtime.need
EXPERIMENT='m2_given_contact_pilot_train'
PREPARATION_SCHEMA='m2_given_eight_pilot_preparation_v1'
PILOT_SCHEMA='m2_given_contact_act_pilot_v1'
WINDOW_COUNTS={'train':3908,'validation':500}
SCHEDULE=dict(experiment=EXPERIMENT,time_limit_seconds=7200,lerobot_root=runtime.VENDOR_ROOT,
    steps=5000,batch_size=64,eval_every=500,seed=20261026,lr=1e-4,weight_decay=1e-4,
    q_noise_std=.002,q_dropout=.05)
FLAGS=dict(diagnostic_only=True,formal_acceptance=False,test_payloads_opened=False,
    training_eligible=False,closed_loop_evaluated=False,automatic_resume=False)


def full_reference(ref):
    runtime._reference_shape(ref)
    need(set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and ref['bytes']>0,
         'Pilot input requires a complete path/bytes/SHA256 reference')
    return ref


def validate_config(config):
    need(set(config)==set(SCHEDULE)|{'base','dataset'} and
         all(config.get(key)==value for key,value in SCHEDULE.items()) and
         Path(str(config.get('base',''))).is_absolute(),
         'One finite 5000-step official CUDA pilot schedule required; no resume')
    full_reference(config['dataset'])
    return config


def load_dataset(ref):
    """Load only hash-bound prepared TRAIN/VAL arrays and manifest metadata."""
    full_reference(ref);complete=runtime.read_bound(ref)
    need(complete.get('schema')==PREPARATION_SCHEMA and complete.get('training_eligible') is False and
         complete.get('test_payloads_opened') is False and complete.get('task_contract')==runtime.task_contract(),
         'Explicit diagnostic preparation/common contract required; no formal qualification')
    refs=complete.get('arrays',{})
    need(set(refs)=={'train','validation'}, 'Only prepared TRAIN/VALIDATION arrays permitted')
    for item in [*refs.values(),complete.get('source_manifest'),complete.get('normalization'),complete.get('index')]:
        full_reference(item)
    manifest=runtime.read_bound(complete['source_manifest'])
    need(manifest.get('schema')==runtime.dataset.MANIFEST_SCHEMA and manifest.get('task_id')==runtime.pose.TASK_ID and
         manifest.get('action_semantics_id')==runtime.given.ACTION_SEMANTICS_ID,
         'Given-contact source manifest required')
    episodes=manifest.get('episodes',[]);need(len(episodes)==8,'Exactly eight pilot expert episodes required')
    episode_map={};scope={}
    for row in episodes:
        runtime.shared.require_membership(row.get('split'),row.get('subject_id'))
        episode_id=row.get('episode_id')
        need(isinstance(episode_id,str) and episode_id and episode_id not in episode_map,
             'Unique pilot episode identifiers required')
        episode_map[episode_id]=row
    for split,n in [('train',7),('validation',1)]:
        selected=[row for row in episodes if row['split']==split]
        identities=sorted({row['subject_id'] for row in selected})
        need(len(selected)==n and len(identities)==n,'Exactly seven TRAIN and one VAL identities required')
        scope[split]=dict(episode_ids=[row['episode_id'] for row in selected],subject_ids=identities,
            episode_count=n,window_count=WINDOW_COUNTS[split])
    need(scope['validation']['subject_ids']==['p001005'],'Frozen eight-case VAL identity must remain p001005')
    index=runtime.read_bound(complete['index'])
    need(index.get('schema')==runtime.dataset.INDEX_SCHEMA and
         index.get('manifest_sha256')==complete['source_manifest']['sha256'] and
         index.get('task_id')==runtime.pose.TASK_ID and
         index.get('action_semantics_id')==runtime.given.ACTION_SEMANTICS_ID,
         'Original prepared given-contact window index required')
    rows=index.get('samples',[])
    need(isinstance(rows,list) and len(rows)==sum(WINDOW_COUNTS.values()),'Complete pilot window index required')
    for row in rows:
        episode=episode_map.get(row.get('episode_id'))
        need(episode is not None and row.get('split')==episode['split'],'Window/episode TRAIN/VAL scope differs')
    prepared={}
    shapes={'states':(17,),'histories':(2,17),'actions':(20,7),'origins':(7,),
        'velocities':(7,),'observation_times':()}
    for split,count in WINDOW_COUNTS.items():
        selected=[row for row in rows if row['split']==split]
        need([row.get('sample_index') for row in selected]==list(range(count)) and
             {row['episode_id'] for row in selected}==set(scope[split]['episode_ids']),
             'Complete original ordered pilot windows required')
        with np.load(runtime.bound_path(refs[split]),allow_pickle=False) as arrays:
            need(set(arrays.files)==set(runtime.shared.ARRAY_KEYS),'Exactly the six approved numerical arrays required')
            samples={key:arrays[key] for key in runtime.shared.ARRAY_KEYS}
        for key,tail in shapes.items():
            value=samples[key]
            need(value.shape==(count,)+tail and value.dtype.kind in 'fiu' and np.isfinite(value).all(),
                 'Finite approved pilot shape required: '+split+'/'+key)
        need(np.array_equal(samples['histories'][:,1],samples['states']) and
             np.array_equal(samples['origins'],samples['states'][:,:7]),
             'Causal history/current state/fixed block origins differ')
        samples.update(schema=runtime.given.DATASET_SCHEMA,task_id=runtime.pose.TASK_ID,
            action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,
            observation_contract=runtime.pose.observation_contract(),handoff_policy_id=runtime.handoff.POLICY_ID)
        prepared[split]=samples
    stats=runtime.validate_normalization(runtime.read_bound(complete['normalization']))
    need(runtime.given.fit_normalization(prepared['train'],split='train')==stats,
         'Pilot normalization must be fitted only on the exact TRAIN arrays')
    prepared.update(normalization=stats,scope=scope,preparation=deepcopy(complete),
        window_order={split:runtime.digest([row for row in rows if row['split']==split]) for split in WINDOW_COUNTS})
    return prepared


def selected_step(rows):
    need(bool(rows),'At least one complete full-VAL evaluation required')
    return min(rows,key=lambda row:(row['validation']['act']['full_chunk_mae_rad'],row['step']))['step']


def checkpoint(policy,prepared,inputs,step,metrics):
    return dict(schema=PILOT_SCHEMA,model=runtime.shared.native_spec(),upstream_commit=runtime.UPSTREAM_COMMIT,
        task_contract=runtime.task_contract(),normalization=prepared['normalization'],scope=prepared['scope'],
        state_only=True,action_representation='block_relative_joint',selection_split='validation',
        inputs=inputs,step=step,metrics=metrics,state_dict=policy.state_dict(),**FLAGS)


def save_checkpoint(torch,path,value):
    with Path(path).open('wb') as stream:torch.save(value,stream)
    return runtime.reference(path)


def fit(config_ref,output):
    started=time.monotonic();output=Path(output).resolve()
    need(not output.exists(),'One fresh pilot output directory required; no resume')
    output.mkdir(parents=True,exist_ok=False)
    policy=None;torch=None;optimizer=None;completed=0;inputs={'config':config_ref};prepared=None
    try:
        config=validate_config(runtime.read_bound(config_ref));prepared=load_dataset(config['dataset'])
        inputs['dataset']=config['dataset']
        # The unchanged original run.py/process_guard owns the process group.
        with runtime.owned_gpu() as ownership:
            identity=runtime._identity()
            need(identity['pid']==identity['pgid']==identity['sid'],'Original guard-owned pilot session required')
            torch,policy,software=runtime.make_model({'lerobot_root':config['lerobot_root']},device='cuda')
            execution=runtime.execution_settings(torch,policy,device='cuda')
            need(sum(parameter.numel() for parameter in policy.parameters())==2658823,
                 'Approved official state-only no-VAE ACT parameter count differs')
            inputs['software']=runtime.write_new(output/'software.json',software)
            runtime.write_new(output/'STARTED.json',dict(schema=PILOT_SCHEMA,inputs=inputs,scope=prepared['scope'],
                execution=execution,ownership=ownership,identity=identity,schedule=SCHEDULE,**FLAGS))
            rng=np.random.Generator(np.random.PCG64(config['seed']))
            augmentation=np.random.Generator(np.random.PCG64(20261027))
            optimizer=torch.optim.AdamW(policy.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],
                betas=(.9,.999),eps=1e-8)
            evaluations=[];best_score=float('inf');last_metrics=None
            with (output/'metrics.jsonl').open('x',buffering=1) as log:
                for step in range(1,config['steps']+1):
                    need(time.monotonic()-started<config['time_limit_seconds'],'Finite two-hour pilot deadline reached')
                    policy.train();ids=rng.integers(len(prepared['train']['states']),size=config['batch_size'])
                    batch=runtime.given.pack_batch(prepared['train'],prepared['normalization'],ids,training=True,
                        rng=augmentation,q_noise_std=config['q_noise_std'],q_dropout=config['q_dropout'])
                    tensors={key:torch.from_numpy(value).to('cuda') for key,value in batch.items()}
                    optimizer.zero_grad(set_to_none=True);loss,_=policy(tensors)
                    need(loss.ndim==0 and bool(torch.isfinite(loss)),'Nonfinite/non-scalar official ACT pilot loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();completed=step
                    if step==1 or step%50==0:
                        row=dict(step=step,train_loss=float(loss.detach().cpu()),grad_norm=float(norm),
                            elapsed_wall_s=time.monotonic()-started)
                        log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    if step%config['eval_every']:continue
                    row={'step':step}
                    for split in ('train','validation'):
                        prediction=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                        row[split]=runtime.metrics(prediction,prepared[split])
                    evaluations.append(row);last_metrics=row
                    log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    score=row['validation']['act']['full_chunk_mae_rad']
                    if score<best_score:
                        best_score=score
                        save_checkpoint(torch,output/'best_pilot.pt',checkpoint(policy,prepared,inputs,step,row))
                last_packet=checkpoint(policy,prepared,inputs,completed,last_metrics)
                last_packet['optimizer_state_dict']=optimizer.state_dict()
                last=save_checkpoint(torch,output/'last_pilot.pt',last_packet)
            selected=selected_step(evaluations)
            best=runtime.reference(output/'best_pilot.pt')
            packet=torch.load(output/'best_pilot.pt',map_location='cpu',weights_only=True)
            need(packet['schema']==PILOT_SCHEMA and packet['step']==selected,'Pilot best checkpoint/earliest VAL argmin differs')
            policy.load_state_dict(packet['state_dict'],strict=True)
            predictions={};selected_metrics={}
            for split in ('train','validation'):
                value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,
                    window_order_sha256=prepared['window_order'][split])
                selected_metrics[split]=runtime.metrics(value,prepared[split])
            complete=dict(schema=PILOT_SCHEMA,complete=True,completed_steps=completed,selected_step=selected,
                best=best,last=last,inputs=inputs,task_contract=runtime.task_contract(),model=runtime.shared.native_spec(),
                scope=prepared['scope'],window_order=prepared['window_order'],evaluations=evaluations,
                selected_metrics=selected_metrics,selected_predictions=predictions,execution=execution,
                ownership=ownership,identity=identity,schedule=SCHEDULE,elapsed_wall_s=time.monotonic()-started,
                training_log=runtime.reference(output/'metrics.jsonl'),**FLAGS)
            runtime.write_new(output/'COMPLETE.json',complete)
            return complete
    except BaseException as error:
        partial=None
        if policy is not None:
            packet=checkpoint(policy,prepared,inputs,completed,None)
            packet.update(complete=False,error=type(error).__name__+': '+str(error))
            if optimizer is not None:packet['optimizer_state_dict']=optimizer.state_dict()
            partial=save_checkpoint(torch,output/'failed_pilot.pt',packet)
        (output/'traceback.txt').write_text(traceback.format_exc())
        runtime.write_new(output/'FAILED.json',dict(schema=PILOT_SCHEMA,complete=False,completed_steps=completed,
            inputs=inputs,partial_checkpoint=partial,error=type(error).__name__+': '+str(error),
            traceback=runtime.reference(output/'traceback.txt'),elapsed_wall_s=time.monotonic()-started,**FLAGS))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=fit(runtime.reference(args.config),args.output)
    print(json.dumps(dict(completed_steps=result['completed_steps'],selected_step=result['selected_step'],**FLAGS)),flush=True)


if __name__=='__main__':main()
