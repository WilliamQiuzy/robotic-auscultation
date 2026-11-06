"""Official state-only ACT on the expanded successful expert TRAIN/VAL set.

Diagnostic only. Raw H20 labels, E10 execution and h2/state17 observations
retain the original contract. No formal population gate, TEST or resume.
"""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_pilot_train as original

need=runtime.need
EXPERIMENT='m2_given_contact_expanded_pilot_train'
MODULE='sim.ausc_policy.'+EXPERIMENT
PREPARATION_SCHEMA='m2_given_diverse_pilot_preparation_v1'
MASKED_PREPARATION_SCHEMA='m2_given_contact_masked_preparation_v1'
PREPARATION_SCHEMAS=(PREPARATION_SCHEMA,MASKED_PREPARATION_SCHEMA)
PILOT_SCHEMA='m2_given_contact_expanded_act_pilot_v1'
SELECTION_SCHEMA='m2_given_contact_expanded_act_selection_v1'
CPU_SCHEMA='m2_given_contact_expanded_act_cpu_equivalence_v1'
SCHEDULE=dict(experiment=EXPERIMENT,time_limit_seconds=21600,lerobot_root=runtime.VENDOR_ROOT,
    steps=20000,batch_size=64,eval_every=500,seed=20261029,lr=1e-4,weight_decay=1e-4,
    q_noise_std=0.,q_dropout=0.)
FLAGS=deepcopy(original.FLAGS)
full_reference=original.full_reference
selected_step=original.selected_step
save_checkpoint=original.save_checkpoint


def validate_config(config):
    variable={'steps','batch_size','eval_every','seed','time_limit_seconds'}
    optional={k for k in ('replacement_pod_gpu_binding','evaluation_cohort') if isinstance(config,dict) and k in config}
    need(isinstance(config,dict) and set(config)==set(SCHEDULE)|{'base','dataset'}|optional and
        all(config.get(k)==v for k,v in SCHEDULE.items() if k not in variable) and
        Path(str(config.get('base',''))).is_absolute(), 'Explicit expanded diagnostic schedule required; no resume')
    for key,low,high in [('steps',1,100000),('batch_size',1,256),('eval_every',1,100000),
                         ('seed',0,2**32-1),('time_limit_seconds',1,86400)]:
        need(type(config.get(key)) is int and low<=config[key]<=high,'Finite integer schedule required: '+key)
    need(config['eval_every']<=config['steps'],'At least one full validation evaluation required')
    full_reference(config['dataset'])
    if 'replacement_pod_gpu_binding' in optional:
        from . import rebuild_gpu_binding
        rebuild_gpu_binding.validate(config['replacement_pod_gpu_binding'],base=config['base'])
        dataset=runtime.read_bound(config['dataset'])
        need(dataset.get('schema') in PREPARATION_SCHEMAS,'Explicit approved dataset version required')
        rebuild_gpu_binding.validate_dataset(dataset,schema=dataset['schema'],task_contract=runtime.task_contract())
    if 'evaluation_cohort' in optional:
        from . import m2_act_recovery_scope
        cohorts=m2_act_recovery_scope.validate(config['evaluation_cohort'])
        dataset=runtime.read_bound(config['dataset'])
        need(dataset.get('schema')==MASKED_PREPARATION_SCHEMA and
             dataset.get('split_subject_ids')==dict(train=sorted(cohorts['train']),validation=sorted(cohorts['selection_validation'])),
             'Pretraining evaluation freeze must remain separate from exact TRAIN/selection data')
    return config


def evaluation_steps(config):
    result=list(range(config['eval_every'],config['steps']+1,config['eval_every']))
    if not result or result[-1]!=config['steps']:result.append(config['steps'])
    return result


def training_protocol(config):
    masked=runtime.read_bound(config['dataset']).get('schema')==MASKED_PREPARATION_SCHEMA
    result=dict(schema='m2_given_contact_expanded_act_protocol_v2' if masked else 'm2_given_contact_expanded_act_protocol_v1',schedule=deepcopy(config),
        model=runtime.shared.native_spec(),parameter_count=2658823,state_only=True,
        observation_dimensions={'state':7,'environment':27,'total':34},
        raw_state_dimensions=17,history_frames=2,action_horizon=20,execution_prefix=10,
        selection='earliest minimum full validation act.full_chunk_mae_rad',
        evaluation_steps=evaluation_steps(config),augmentation={'q_noise_std':0.,'q_dropout':0.},
        model_dropout=.1,task_contract=runtime.task_contract(),**FLAGS)
    if masked:result.update(label_protocol=runtime.shared.MASKED_LABEL_PROTOCOL,
        masked_suffix_is_supervision=False,normalization_and_metrics='real future cells only',
        training_sampling='half uniform real TRAIN origins, half real TRAIN origins within 60mm of prescribed pose; no state or action noise')
    return result


def training_indices(rng,samples,size):
    if samples.get('label_protocol')!=runtime.shared.MASKED_LABEL_PROTOCOL:
        return rng.integers(len(samples['states']),size=size)
    near=np.flatnonzero(np.linalg.norm(samples['states'][:,7:10],axis=1)<=.060)
    need(len(near)>0,'Real near-goal TRAIN coverage required before masked training')
    n=size//2
    return np.r_[near[rng.integers(len(near),size=n)],rng.integers(len(samples['states']),size=size-n)]


def code_binding():
    root=Path(__file__).resolve().parents[2]
    modules=(Path(__file__).name,'m2_given_contact_pilot_train.py','m2_given_contact_runtime.py',
        'm2_act.py','m2_given_contact_act.py','m2_given_contact_pose.py','m2_given_contact_handoff.py','rebuild_gpu_binding.py',
        'm2_act_recovery_scope.py')
    commit_file=root/'CODE_COMMIT'
    commit=commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(
        ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return dict(code_commit=commit,files={name:runtime.reference(root/'sim/ausc_policy'/name) for name in modules})


def validate_code(binding):
    current=code_binding()
    need(set(binding.get('files',{}))==set(current['files']) and all(
        {k:ref[k] for k in ('bytes','sha256')}=={k:current['files'][name][k] for k in ('bytes','sha256')}
        for name,ref in binding['files'].items()),'Expanded trained numerical/protocol source bytes changed')


def load_dataset(ref):
    full_reference(ref);complete=runtime.read_bound(ref)
    need(complete.get('schema') in PREPARATION_SCHEMAS and complete.get('complete') is True and
        complete.get('diagnostic_only') is True and complete.get('formal_acceptance') is False and
        complete.get('training_eligible') is False and complete.get('test_payloads_opened') is False and
        complete.get('task_contract')==runtime.task_contract(), 'Explicit complete expanded diagnostic preparation required')
    counts=complete.get('counts',{});refs=complete.get('arrays',{})
    need(set(counts)==set(refs)=={'train','validation'} and all(type(n) is int and n>0 for n in counts.values()),
        'Positive variable TRAIN/VAL counts and arrays required')
    episodes=complete.get('episodes');need(isinstance(episodes,list) and episodes,'Successful expert episode metadata required')
    episode_map={};scope={}
    for episode in episodes:
        runtime.shared.require_membership(episode.get('split'),episode.get('subject_id'))
        eid=episode.get('episode_id')
        need(isinstance(eid,str) and eid and eid not in episode_map and
            type(episode.get('windows')) is int and episode['windows']>0,'Unique positive-window expert episodes required')
        episode_map[eid]=episode
    for split,count in counts.items():
        selected=[row for row in episodes if row['split']==split]
        subjects=sorted({row['subject_id'] for row in selected})
        need(selected and complete.get('split_subject_ids',{}).get(split)==subjects,
            'Exact expanded TRAIN/VAL identity inventory required')
        cursor=0
        for row in selected:
            need(row.get('split_sample_range')==[cursor,cursor+row['windows']], 'Complete contiguous episode sample ranges required')
            cursor+=row['windows']
        need(cursor==count,'Expanded episode windows/count differs')
        scope[split]=dict(episode_ids=[row['episode_id'] for row in selected],subject_ids=subjects,
            episode_count=len(selected),window_count=count)
    need(not set(scope['train']['subject_ids'])&set(scope['validation']['subject_ids']), 'TRAIN/VAL identities must be disjoint')
    for item in [*refs.values(),complete.get('normalization'),complete.get('index')]:full_reference(item)
    index=runtime.read_bound(complete['index'])
    need(index.get('schema')==runtime.dataset.INDEX_SCHEMA and index.get('task_id')==runtime.pose.TASK_ID and
        index.get('action_semantics_id')==runtime.given.ACTION_SEMANTICS_ID,'Expanded causal window index required')
    rows=index.get('samples',[])
    need(isinstance(rows,list) and len(rows)==sum(counts.values()),'Complete expanded window count/index required')
    for row in rows:
        episode=episode_map.get(row.get('episode_id'))
        need(episode is not None and row.get('split')==episode['split'], 'Expanded window/episode TRAIN/VAL scope differs')
    masked=complete['schema']==MASKED_PREPARATION_SCHEMA
    array_keys=runtime.shared.ARRAY_KEYS+(('action_is_pad',) if masked else ())
    prepared={};shapes={'states':(17,),'histories':(2,17),'actions':(20,7),'origins':(7,),
        'velocities':(7,),'observation_times':()}
    for split,count in counts.items():
        selected=[row for row in rows if row['split']==split]
        need([row.get('sample_index') for row in selected]==list(range(count)), 'Complete ordered expanded windows required')
        for episode in [e for e in episodes if e['split']==split]:
            start,end=episode['split_sample_range']
            need(all(row['episode_id']==episode['episode_id'] for row in selected[start:end]),
                'Expanded window/episode sample range differs')
        with np.load(runtime.bound_path(refs[split]),allow_pickle=False) as arrays:
            need(set(arrays.files)==set(array_keys),'Exact versioned numerical array inventory required')
            samples={key:arrays[key] for key in array_keys}
        for key,tail in shapes.items():
            value=samples[key]
            need(value.shape==(count,)+tail and value.dtype.kind in 'fiu' and np.isfinite(value).all(),
                'Finite approved expanded shape required: '+split+'/'+key)
        need(np.array_equal(samples['histories'][:,1],samples['states']) and
            np.array_equal(samples['origins'],samples['states'][:,:7]),'Raw causal state/history/fixed origins differ')
        samples.update(schema=runtime.given.MASKED_DATASET_SCHEMA if masked else runtime.given.DATASET_SCHEMA,task_id=runtime.pose.TASK_ID,
            action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,observation_contract=runtime.pose.observation_contract(),
            handoff_policy_id=runtime.handoff.POLICY_ID)
        if masked:samples['label_protocol']=runtime.shared.MASKED_LABEL_PROTOCOL
        runtime.given.require_samples(samples)
        prepared[split]=samples
    stats=runtime.validate_normalization(runtime.read_bound(complete['normalization']))
    need(runtime.given.fit_normalization(prepared['train'],split='train')==stats,
        'Expanded normalization must be fitted on exact TRAIN arrays only')
    prepared.update(normalization=stats,scope=scope,preparation=deepcopy(complete),
        window_order={split:runtime.digest([row for row in rows if row['split']==split]) for split in counts})
    return prepared


def checkpoint(policy,prepared,inputs,step,metrics):
    result=original.checkpoint(policy,prepared,inputs,step,metrics)
    result.update(schema=PILOT_SCHEMA,protocol=training_protocol(runtime.read_bound(inputs['config'])))
    return result


def load_selected_policy(selection):
    """Actual selected weights only; no optimizer or expert action fallback."""
    need(selection.get('schema') in (PILOT_SCHEMA,SELECTION_SCHEMA) and
        all(selection.get(k) is v for k,v in FLAGS.items()), 'Expanded diagnostic selected weights required')
    config=validate_config(runtime.read_bound(selection['inputs']['config']))
    need(selection.get('protocol')==training_protocol(config) and selection['inputs']['dataset']==config['dataset'],
        'Expanded exact training/data/protocol binding differs')
    validate_code(selection['inputs']['code'])
    torch,policy,software=runtime.make_model({'lerobot_root':config['lerobot_root']},device='cpu')
    saved=runtime.read_bound(selection['inputs']['software'])
    for key in ('vendor','dependencies','python','prefix','python_version'):
        need(saved[key]==software[key],'Official expanded CPU environment changed: '+key)
    need({Path(k).name:v for k,v in saved['source_bindings'].items()}==
        {Path(k).name:v for k,v in software['source_bindings'].items()},'Expanded trained numerical source bytes changed')
    packet=torch.load(io.BytesIO(runtime.bound_path(selection['best']).read_bytes()),map_location='cpu',weights_only=True)
    need(packet.get('schema')==PILOT_SCHEMA and packet.get('step')==selection['selected_step'] and
        packet.get('inputs')==selection['inputs'] and packet.get('scope')==selection['scope'] and
        packet.get('protocol')==selection['protocol'] and packet.get('task_contract')==runtime.task_contract() and
        packet.get('model')==runtime.shared.native_spec() and all(packet.get(k) is v for k,v in FLAGS.items()),
        'Exact selected expanded checkpoint required')
    runtime.validate_normalization(packet['normalization']);policy.load_state_dict(packet['state_dict'],strict=True);policy.eval()
    return torch,policy,packet


def verify_selected(selection_ref,receipt_path):
    selection=runtime.read_bound(selection_ref);prepared=load_dataset(selection['inputs']['dataset'])
    need(prepared['scope']==selection['scope'] and prepared['window_order']==selection['window_order'],
        'Selected expanded scope/window ordering changed')
    torch,policy,packet=load_selected_policy(selection);execution=runtime.execution_settings(torch,policy,device='cpu')
    comparisons={};metrics={}
    for split in ('train','validation'):
        cpu=runtime.infer(torch,policy,prepared[split],packet['normalization'],device='cpu')
        cuda=runtime.load_predictions(selection['selected_predictions'][split],window_order_sha256=prepared['window_order'][split])
        metrics[split]=runtime.metrics(cpu,prepared[split])
        comparisons[split]=runtime.compare_devices(cpu,cuda,metrics[split],selection['selected_metrics'][split])
    return runtime.write_new(receipt_path,dict(schema=CPU_SCHEMA,passed=True,selection=selection_ref,
        checkpoint=selection['best'],selected_step=selection['selected_step'],scope=selection['scope'],
        comparisons=comparisons,metrics=metrics,execution=execution,**FLAGS))


def fit(config_ref,output):
    started=time.monotonic();output=Path(output).resolve();need(not output.exists(),'Fresh expanded output required; no resume')
    output.mkdir(parents=True,exist_ok=False)
    policy=None;torch=None;optimizer=None;completed=0;inputs={'config':config_ref};prepared=None
    try:
        config=validate_config(runtime.read_bound(config_ref));prepared=load_dataset(config['dataset'])
        inputs.update(dataset=config['dataset'],code=code_binding());protocol=training_protocol(config)
        lease=(runtime.owned_gpu(config['replacement_pod_gpu_binding']) if 'replacement_pod_gpu_binding' in config else runtime.owned_gpu())
        with lease as ownership:
            identity=runtime._identity();need(identity['pid']==identity['pgid']==identity['sid'],'Original guard-owned expanded session required')
            torch,policy,software=runtime.make_model({'lerobot_root':config['lerobot_root']},device='cuda')
            execution=runtime.execution_settings(torch,policy,device='cuda')
            need(sum(p.numel() for p in policy.parameters())==2658823,'Approved official state-only ACT parameter count differs')
            inputs['software']=runtime.write_new(output/'software.json',software)
            runtime.write_new(output/'STARTED.json',dict(schema=PILOT_SCHEMA,inputs=inputs,scope=prepared['scope'],
                protocol=protocol,execution=execution,ownership=ownership,identity=identity,**FLAGS))
            rng=np.random.Generator(np.random.PCG64(config['seed']))
            optimizer=torch.optim.AdamW(policy.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8)
            evaluations=[];best_score=float('inf');last_metrics=None;eval_steps=set(evaluation_steps(config))
            with (output/'metrics.jsonl').open('x',buffering=1) as log:
                for step in range(1,config['steps']+1):
                    need(time.monotonic()-started<config['time_limit_seconds'],'Finite expanded training deadline reached')
                    policy.train();ids=training_indices(rng,prepared['train'],config['batch_size'])
                    batch=runtime.given.pack_batch(prepared['train'],prepared['normalization'],ids,training=True,
                        rng=rng,q_noise_std=0.,q_dropout=0.)
                    tensors={key:torch.from_numpy(value).to('cuda') for key,value in batch.items()}
                    optimizer.zero_grad(set_to_none=True);loss,_=policy(tensors)
                    need(loss.ndim==0 and bool(torch.isfinite(loss)),'Finite scalar official expanded ACT loss required')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();completed=step
                    if step==1 or step%50==0:
                        row=dict(step=step,train_loss=float(loss.detach().cpu()),grad_norm=float(norm),elapsed_wall_s=time.monotonic()-started)
                        log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    if step not in eval_steps:continue
                    row={'step':step}
                    for split in ('train','validation'):
                        prediction=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                        row[split]=runtime.metrics(prediction,prepared[split])
                    evaluations.append(row);last_metrics=row
                    log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    score=row['validation']['act']['full_chunk_mae_rad']
                    if score<best_score:
                        best_score=score;save_checkpoint(torch,output/'best_expanded_pilot.pt',checkpoint(policy,prepared,inputs,step,row))
                last_packet=checkpoint(policy,prepared,inputs,completed,last_metrics);last_packet['optimizer_state_dict']=optimizer.state_dict()
                last=save_checkpoint(torch,output/'last_expanded_pilot.pt',last_packet)
            selected=selected_step(evaluations);best=runtime.reference(output/'best_expanded_pilot.pt')
            packet=torch.load(output/'best_expanded_pilot.pt',map_location='cpu',weights_only=True)
            need(packet['schema']==PILOT_SCHEMA and packet['step']==selected,'Earliest full-VAL selected checkpoint differs')
            policy.load_state_dict(packet['state_dict'],strict=True);predictions={};selected_metrics={}
            for split in ('train','validation'):
                value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,window_order_sha256=prepared['window_order'][split])
                selected_metrics[split]=runtime.metrics(value,prepared[split])
            selection=dict(schema=SELECTION_SCHEMA,selected_step=selected,best=best,inputs=inputs,scope=prepared['scope'],
                window_order=prepared['window_order'],selected_predictions=predictions,selected_metrics=selected_metrics,protocol=protocol,**FLAGS)
            selection_ref=runtime.write_new(output/'SELECTED.json',selection)
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            subprocess.run([sys.executable,'-B','-m',MODULE,'--verify-selected',selection_ref['path'],
                '--receipt',str(output/'CPU_EQUIVALENCE.json')],check=True,env=environment,timeout=1800)
            cpu=runtime.reference(output/'CPU_EQUIVALENCE.json');need(runtime.read_bound(cpu).get('passed') is True,'Actual CPU equivalence required')
            complete=dict(selection,schema=PILOT_SCHEMA,complete=True,completed_steps=completed,last=last,
                task_contract=runtime.task_contract(),model=runtime.shared.native_spec(),evaluations=evaluations,
                cpu_equivalence=cpu,execution=execution,ownership=ownership,identity=identity,schedule=deepcopy(config),
                elapsed_wall_s=time.monotonic()-started,training_log=runtime.reference(output/'metrics.jsonl'))
            runtime.write_new(output/'COMPLETE.json',complete);return complete
    except BaseException as error:
        partial=None
        if policy is not None:
            packet=checkpoint(policy,prepared,inputs,completed,None);packet.update(complete=False,error=repr(error))
            if optimizer is not None:packet['optimizer_state_dict']=optimizer.state_dict()
            partial=save_checkpoint(torch,output/'failed_expanded_pilot.pt',packet)
        (output/'traceback.txt').write_text(traceback.format_exc())
        runtime.write_new(output/'FAILED.json',dict(schema=PILOT_SCHEMA,complete=False,completed_steps=completed,
            inputs=inputs,partial_checkpoint=partial,error=repr(error),traceback=runtime.reference(output/'traceback.txt'),**FLAGS))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--output');parser.add_argument('--verify-selected');parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.verify_selected:
        need(args.receipt and not args.config and not args.output,'Explicit CPU verification receipt required')
        print(json.dumps(verify_selected(runtime.reference(args.verify_selected),args.receipt)),flush=True)
    else:
        need(args.config and args.output,'Expanded training config and fresh output required')
        result=fit(runtime.reference(args.config),args.output)
        print(json.dumps(dict(completed_steps=result['completed_steps'],selected_step=result['selected_step'],**FLAGS)),flush=True)


if __name__=='__main__':main()
