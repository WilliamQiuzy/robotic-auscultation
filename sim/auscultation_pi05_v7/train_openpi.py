"""Pinned pi05 delta-only ablation, with actual sampled-action diagnostics.

The immutable V6 qualified data retains its original train/validation/test
assignment. This pilot is diagnostic; it cannot certify generalization or Isaac.
"""
import argparse
from collections import deque
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from sim.auscultation_pi05_v6 import train_openpi as parent
from sim.auscultation_pi05_v6.evaluation import action_metrics,balanced_indices
from sim.auscultation_pi05_v6.training_data import phase_balanced_indices
from sim.auscultation_pi05_v6.action_contract import bounded
from .training_data import load_dataset,make_batch,decode_prediction,core_sources,RECIPE
from . import model as implementation

def utc():return datetime.now(timezone.utc).isoformat()

def sampled_diagnostic(dataset,runtime,sample,step,output,*,per_phase=2):
    import jax
    from openpi.models import model as model_api
    report=dict(schema='wam_pi05_v7_sampled_diagnostic_v1',step=step,seed=917,sampling_steps=10,
        data_hash=dataset.data_hash,scope='Seen train and inspected held-out target plans, one body/layout; not fresh generalization or physics',
        closed_loop_validated=False,episodes={})
    predictions={}
    for ei,episode in enumerate(dataset.episodes):
        split=episode['split']
        candidates=[(index,start) for index,(e,start) in enumerate(dataset.indices[split]) if e==ei]
        selected=balanced_indices([str(episode['phases'][start]) for _,start in candidates],per_phase=per_phase)
        pred=[];ref=[];rows=[];hold=[];normalized=[]
        for choice in selected:
            index,start=candidates[choice]
            obs,_=make_batch(dataset,split,[index],runtime['tokenizer'])
            result=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
            if result.shape!=(1,10,7) or not np.isfinite(result).all():raise ValueError('Invalid sampled action')
            phase=str(episode['phases'][start]);state=episode['state'][start]
            pred.append(decode_prediction(dataset,result[0],state,phase));normalized.append(result[0])
            ref.append(episode['actions'][start:start+10]);hold.append(np.repeat(state[None,:7],10,axis=0))
            rows.append(dict(frame=start,phase=phase,target=str(episode['targets'][start])))
        pred,ref,hold=np.asarray(pred),np.asarray(ref),np.asarray(hold)
        key=episode['episode_id']
        predictions[key+'_prediction']=pred;predictions[key+'_reference']=ref
        predictions[key+'_normalized']=np.asarray(normalized)
        report['episodes'][key]=dict(split=split,records=rows,metrics=action_metrics(pred,ref),
            measured_hold_baseline=action_metrics(hold,ref),
            first_action_bound_pass_fraction=float(np.mean([bounded(p[:1],r['phase']) for p,r in zip(pred,rows)])),
            full_chunk_bound_pass_fraction=float(np.mean([bounded(p,r['phase']) for p,r in zip(pred,rows)])),
            by_phase={phase:action_metrics(pred[np.array([r['phase']==phase for r in rows])],ref[np.array([r['phase']==phase for r in rows])]) for phase in sorted({r['phase'] for r in rows})})
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'predictions.npz',**predictions)
    report['predictions_sha256']=parent._sha(output/'predictions.npz')
    parent._atomic_json(output/'report.json',report)
    return report

def run(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from openpi.models import model as model_api
    output=args.output.resolve()
    if output.exists() and not args.resume:raise ValueError('Fresh output or explicit resume required')
    output.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset(args.manifest)
    runtime=implementation.initialize(args.seed,args.learning_rate)
    config=dict(recipe=RECIPE,source_sha256=core_sources(),trainer_sha256=parent._sha(__file__),
        parent_data_hash=dataset.parent.data_hash,openpi_commit=parent.OPENPI_COMMIT,assets=runtime['assets'],
        seed=args.seed,learning_rate=args.learning_rate,batch_size=args.batch_size,
        action_dim=7,action_horizon=10,state_width=41,max_token_len=256,
        total_steps=args.steps,evaluate_every=args.evaluate_every,save_every=args.save_every,
        model_architecture='unmodified official pi05 with V6 seven-channel LoRA/projection partition',
        action_semantics='absolute_joint_command_v1',model_action_semantics='future_command_minus_fixed_observation_measured_q',
        canonical_phase_instructions=dict(parent.INSTRUCTIONS),instruction_policy='canonical-phase-and-target-v1',
        observation_contract=dataset.manifest['observation_contract'],
        action_encoding_binding=dataset.normalization['action_encoding_binding'],
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256'])
    stop=[False]
    def signal_stop(signum,frame):stop[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
    sampler=np.random.default_rng(args.seed);rng=jax.random.key(args.seed);step=0
    if args.resume:
        meta,saved_rng=implementation.restore(runtime,args.resume,dataset.data_hash,config,dataset.normalization)
        step=meta['step'];sampler.bit_generator.state=meta['sampler_rng_state'];rng=jax.random.wrap_key_data(saved_rng)
    parent._atomic_json(output/'configuration.json',dict(config=config,data_hash=dataset.data_hash,
        normalization=dataset.normalization,manifest=str(args.manifest.resolve()),manifest_sha256=parent._sha(args.manifest)))
    def objective(parameters,frozen,obs,actions,key):
        model=nnx.merge(runtime['graph'],parameters,frozen);model.train()
        return model.compute_loss(key,obs,actions,train=True).mean()
    gradient=jax.jit(jax.value_and_grad(objective,argnums=0))
    @jax.jit
    def update(parameters,state,grads):
        updates,state=runtime['optimizer'].update(grads,state,parameters)
        return optax.apply_updates(parameters,updates),state
    sample=implementation.sampler(runtime)
    phases=[str(dataset.episodes[e]['phases'][i]) for e,i in dataset.indices['train']]
    episodes=[e for e,i in dataset.indices['train']]
    metrics_path=output/'metrics.jsonl';history=deque(maxlen=50)
    receipt=dict(schema='wam_pi05_v7_training_run_v1',started_utc=utc(),pid=os.getpid(),
        data_hash=dataset.data_hash,parent_data_hash=dataset.parent.data_hash,resume=None if not args.resume else str(args.resume),
        requested_total_steps=args.steps,requested_stop_at=args.stop_at,step=step,training_complete=False,closed_loop_validated=False)
    def save_receipt():parent._atomic_json(output/'training_run.json',receipt)
    def save_checkpoint():
        leaves=jax.tree.leaves((runtime['trainable'],runtime['opt_state']))
        arrays={'leaf_%05d'%i:np.asarray(v) for i,v in enumerate(jax.device_get(leaves))}
        arrays['jax_rng']=np.asarray(jax.random.key_data(rng))
        metadata=dict(data_hash=dataset.data_hash,config=config,normalization=dataset.normalization,
            sampler_rng_state=sampler.bit_generator.state,leaf_contract=runtime['leaf_contract'],
            best_step=None,best_validation_loss=None,heldout_axes=dataset.manifest['heldout_axes'])
        return parent.save_checkpoint(output/'model',step=step,arrays=arrays,metadata=metadata)
    save_receipt();last_saved=step
    try:
        while step<min(args.steps,args.stop_at) and not stop[0]:
            indices=phase_balanced_indices(phases,args.batch_size,sampler,episodes)
            obs,actions=make_batch(dataset,'train',indices,runtime['tokenizer'])
            observation=model_api.Observation(**jax.device_put(obs));rng,key=jax.random.split(rng)
            loss,grads=gradient(runtime['trainable'],runtime['frozen'],observation,jax.device_put(actions),key)
            if not bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(g).all() for g in jax.tree.leaves(grads)])))):
                raise ValueError('Nonfinite gradient')
            value=float(loss)
            if not np.isfinite(value):raise ValueError('Nonfinite loss')
            runtime['trainable'],runtime['opt_state']=update(runtime['trainable'],runtime['opt_state'],grads)
            jax.block_until_ready((runtime['trainable'],runtime['opt_state']))
            step+=1;history.append(value)
            metrics=dict(step=step,train_loss=value,mean_last_50=float(np.mean(history)))
            if step%args.save_every==0 or step%args.evaluate_every==0 or step==min(args.steps,args.stop_at) or stop[0]:
                checkpoint=save_checkpoint();last_saved=step;receipt['latest_checkpoint']=str(checkpoint)
            if step%args.evaluate_every==0 or step==min(args.steps,args.stop_at):
                result=sampled_diagnostic(dataset,runtime,sample,step,output/('sampled_step_%06d'%step))
                metrics['sampled_first_mae_deg']={name:float(np.degrees(v['metrics']['first_action_joint_mae_rad'])) for name,v in result['episodes'].items()}
                metrics['sampled_first_bound_fraction']={name:v['first_action_bound_pass_fraction'] for name,v in result['episodes'].items()}
            with metrics_path.open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step%25==0 or 'sampled_first_mae_deg' in metrics:print(json.dumps(metrics),flush=True)
            receipt.update(step=step,last_metrics=metrics);save_receipt()
        if step>last_saved:receipt['latest_checkpoint']=str(save_checkpoint())
        receipt.update(training_complete=step==args.steps,requested_segment_complete=step==min(args.steps,args.stop_at),interrupted=stop[0])
    except BaseException as exc:
        receipt['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        receipt['finished_utc']=utc();save_receipt()
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--stop-at',type=int,default=4000);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=61);p.add_argument('--learning-rate',type=float,default=5e-5)
    p.add_argument('--save-every',type=int,default=250);p.add_argument('--evaluate-every',type=int,default=250)
    args=p.parse_args()
    if min(args.steps,args.stop_at,args.batch_size,args.save_every,args.evaluate_every)<1 or args.stop_at>args.steps:
        p.error('Positive training limits required and stop-at must not exceed total steps')
    if args.seed<0 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:p.error('Invalid seed or rate')
    with (parent.ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU workload is active')
        print(json.dumps(run(args),indent=2),flush=True)

if __name__=='__main__':main()
