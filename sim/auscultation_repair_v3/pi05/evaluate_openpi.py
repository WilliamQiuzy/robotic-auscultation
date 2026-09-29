"""Sample a completed task checkpoint on frozen held-out observations.

This is offline evaluation only. No action is sent to Isaac or to hardware.
All interventions share the same per-window sampling noise. Depth enters via
tracked geometry; ablation measures channel dependence, not clinical efficacy.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import train_openpi as training
from .evaluation import action_metrics,balanced_indices,intervene_state,STATE_INTERVENTIONS,mask_inactive_force,training_phase_means
from .openpi_adapter import OPENPI_COMMIT,trainable_path,tokenize_state
from .training_data import denormalize
from .action_contract import bounded


def run(manifest,checkpoint,output,*,per_phase=5,sampling_steps=10,seed=9841):
    repo=training.ROOT/'vendor/openpi'
    if Path(sys.prefix).resolve()!=(repo/'.venv').resolve():
        raise RuntimeError('official openpi environment required')
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    import sentencepiece
    from openpi.models import model as model_api
    from openpi.models.pi0_config import Pi0Config
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':
        raise RuntimeError('one GPU required for task model evaluation')
    training.verify_openpi_checkout(repo,inspect.getfile(Pi0Config))
    dataset=training.load_dataset(manifest)
    checkpoint=Path(checkpoint);output=Path(output)
    if output.exists():raise ValueError('fresh evaluation output required')
    metadata=training._read_json(checkpoint/'metadata.json');config=metadata['config']
    if config.get('state_encoding_binding')!=dataset.normalization['encoding_binding']:
        raise ValueError('state encoding source binding changed')
    if metadata['step']<=0 or config['openpi_commit']!=OPENPI_COMMIT:
        raise ValueError('completed pinned task optimization required')
    for key,name in [('trainer_sha256','train_openpi.py'),('adapter_sha256','openpi_adapter.py'),('training_data_sha256','training_data.py'),('action_contract_sha256','action_contract.py')]:
        if config[key]!=training._sha(Path(__file__).with_name(name)):
            raise ValueError('training implementation hash changed: '+name)
    arrays,metadata=training.load_checkpoint(checkpoint,data_hash=dataset.data_hash,
        config=config,normalization=dataset.normalization)
    parameters_path,tokenizer_path,assets=training._verify_public_assets(training.ROOT)
    if assets!=config['assets']:raise ValueError('public asset binding changed')
    tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    cfg=Pi0Config(pi05=True,action_dim=7,action_horizon=10,max_token_len=200,
        paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    skeleton=nnx.eval_shape(cfg.create,jax.random.key(config['seed']))
    reference=traverse_util.flatten_dict(nnx.state(skeleton).to_pure_dict())
    pretrained=traverse_util.flatten_dict(model_api.restore_params(parameters_path,dtype=jnp.bfloat16))
    path_ids={path:i for i,path in enumerate(reference)}
    merged=training.merge_pretrained(reference,pretrained,
        lambda path,shape:jax.random.normal(jax.random.fold_in(jax.random.key(config['seed']),path_ids[path]),shape,jnp.float32)*.01)
    model=cfg.load(traverse_util.unflatten_dict(merged))
    graph,trainable,frozen=nnx.split(model,nnx.All(nnx.Param,lambda path,value:trainable_path(path)),...)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(config['learning_rate']))
    tree=(trainable,optimizer.init(trainable))
    path_leaves,definition=jax.tree_util.tree_flatten_with_path(tree)
    contract=[dict(path=jax.tree_util.keystr(path),shape=list(value.shape),dtype=str(value.dtype)) for path,value in path_leaves]
    if contract!=metadata['leaf_contract'] or set(arrays)!=({'leaf_%05d'%i for i in range(len(path_leaves))}|{'jax_rng'}):
        raise ValueError('task checkpoint tree differs from runtime')
    leaves=[]
    for i,(_,expected) in enumerate(path_leaves):
        value=arrays['leaf_%05d'%i]
        if value.shape!=expected.shape or str(value.dtype)!=str(expected.dtype):
            raise ValueError('checkpoint leaf shape/dtype mismatch')
        leaves.append(jnp.asarray(value))
    trainable,_=jax.tree_util.tree_unflatten(definition,leaves)
    del skeleton,pretrained,merged,model,tree,path_leaves,arrays,leaves
    @jax.jit
    def sample(parameters,frozen_parameters,observation,rng):
        candidate=nnx.merge(graph,parameters,frozen_parameters);candidate.eval()
        return candidate.sample_actions(rng,observation,num_steps=sampling_steps)
    output.mkdir(parents=True)
    variants=['baseline','rgb',*STATE_INTERVENTIONS,'instruction']
    phase_means=training_phase_means([e for e in dataset.episodes if e['split']=='train'])
    report=dict(schema='wam_pi05_offline_evaluation_v1',data_hash=dataset.data_hash,
        evaluator_sha256=training._sha(__file__),evaluation_helpers_sha256=training._sha(Path(__file__).with_name('evaluation.py')),
        checkpoint=str(checkpoint),checkpoint_metadata_sha256=training._sha(checkpoint/'metadata.json'),
        checkpoint_step=metadata['step'],task_training=True,closed_loop_validated=False,
        sampling_steps=sampling_steps,seed=seed,per_phase=per_phase,
        scope='Diagnostic comparison on previously inspected target plans in one body/layout; not a fresh test, unseen-person or patient-motion claim',
        split_limit='Target plans were solved independently, but canonical transit prefixes can overlap across splits; approach/contact phase errors are more informative than aggregate error.',
        intervention_scope='Direct normalized channel groups, blank RGB, or generic instruction; off-manifold dependence probes. Derived geometry retains information from Hall and joints, so these are not complete upstream-sensor removals.',splits={})
    for split in ('validation','test'):
        if not dataset.indices[split]:continue
        phases=[str(dataset.episodes[e]['phases'][start]) for e,start in dataset.indices[split]]
        selected=balanced_indices(phases,per_phase=per_phase)
        predictions={key:[] for key in variants};references=[];records=[];sample_seconds=[]
        for ordinal,index in enumerate(selected):
            base,actions=training.make_batch(dataset,split,[index],tokenizer)
            e,start=dataset.indices[split][index];episode=dataset.episodes[e]
            phase=str(episode['phases'][start]);target=str(episode['targets'][start])
            references.append(denormalize(actions[0],dataset.normalization['actions']))
            rng=jax.random.fold_in(jax.random.key(seed),ordinal)
            for variant in variants:
                observation=dict(base)
                if variant=='rgb':
                    observation['images']=dict(base['images']);observation['images']['base_0_rgb']=np.zeros_like(base['images']['base_0_rgb'])
                if variant in STATE_INTERVENTIONS or variant=='instruction':
                    state=base['state'][0].copy()
                    if variant in STATE_INTERVENTIONS:state=intervene_state(state,variant)
                    prompt=('Perform the requested robot task.' if variant=='instruction' else training.training_instruction(phase,target))
                    tokens,mask=tokenize_state(tokenizer,prompt,state)
                    observation.update(state=state[None].astype(np.float32),tokenized_prompt=tokens[None],tokenized_prompt_mask=mask[None])
                started=time.monotonic()
                result=np.asarray(sample(trainable,frozen,model_api.Observation(**jax.device_put(observation)),rng))
                sample_seconds.append(dict(variant=variant,ordinal=ordinal,seconds=time.monotonic()-started))
                if result.shape!=(1,10,7) or not np.isfinite(result).all():raise ValueError('invalid sampled action')
                predictions[variant].append(denormalize(result[0],dataset.normalization['actions']))
            records.append(dict(dataset_index=index,episode_id=episode['episode_id'],frame=start,phase=phase,target=target))
            if ordinal%5==0:print(json.dumps(dict(split=split,completed=ordinal+1,total=len(selected))),flush=True)
        reference=np.asarray(references);predictions={key:np.asarray(value) for key,value in predictions.items()}
        constants=np.asarray([np.repeat(phase_means[r['phase']][None],10,axis=0) for r in records])
        zero_motion=np.zeros_like(reference)
        for i,record in enumerate(records):
            if record['phase'] in {'CONTACT','HOLD'}:zero_motion[i,:,6]=4.
        np.savez_compressed(output/(split+'_predictions.npz'),reference=reference,**predictions)
        report['splits'][split]=dict(records=records,metrics={key:action_metrics(value,reference) for key,value in predictions.items()},
            descriptive_baselines={'training_phase_mean':action_metrics(constants,reference),'zero_motion_stage_force':action_metrics(zero_motion,reference)},
            descriptive_baselines_by_phase={phase:{name:action_metrics(value[np.array([r['phase']==phase for r in records])],reference[np.array([r['phase']==phase for r in records])]) for name,value in [('training_phase_mean',constants),('zero_motion_stage_force',zero_motion)]} for phase in sorted(set(r['phase'] for r in records))},
            descriptive_baseline_scope='Additional descriptive comparisons defined before model evaluation: train-only phase mean, or zero pose offset with the supervisor stage force reference. Neither is an executable controller or a checkpoint selection criterion.',
            baseline_by_phase={phase:action_metrics(predictions['baseline'][np.array([r['phase']==phase for r in records])],reference[np.array([r['phase']==phase for r in records])]) for phase in sorted(set(r['phase'] for r in records))},
            intervention_prediction_change={key:action_metrics(value,predictions['baseline']) for key,value in predictions.items() if key!='baseline'},
            proposal_bound_pass_fraction={key:float(np.mean([bounded(chunk,record['phase']) for chunk,record in zip(value,records)])) for key,value in predictions.items()},
            bound_pass_fraction_after_inactive_force_mask={key:float(np.mean([bounded(mask_inactive_force(chunk,record['phase']),record['phase']) for chunk,record in zip(value,records)])) for key,value in predictions.items()},
            inactive_force_mask_scope='Force channel set to zero only in FREE/APPROACH/RETRACT; no active force or motion clipping; this is not an execution success rate',
            sample_seconds=sample_seconds,sampling_timing_scope='Includes first compile; subsequent calls still exclude perception, planning and execution',
            predictions_sha256=training._sha(output/(split+'_predictions.npz')))
    if not report['splits']:raise ValueError('no held-out observations')
    training._atomic_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.manifest,a.checkpoint,a.output),indent=2,allow_nan=False))


if __name__=='__main__':main()
