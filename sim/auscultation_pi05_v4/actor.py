"""Hash-verified pi05 inference, identical training preprocessing."""
import inspect
from pathlib import Path
import sys
import time
import numpy as np
from . import train_openpi as training
from .openpi_adapter import OPENPI_COMMIT,trainable_path,tokenize_state,prepare_images
from .training_data import normalize,denormalize,action_stats


def load_actor(manifest,checkpoint,*,sampling_steps=10,seed=917):
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
    checkpoint=Path(checkpoint)
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
    def predict(rgb,state,phase,target):
        state=normalize(np.asarray(state,dtype=np.float32),dataset.normalization['state'])
        tokens,mask=tokenize_state(tokenizer,training.training_instruction(phase,target),state)
        pixels,present=prepare_images(rgb)
        observation=model_api.Observation(images={k:v[None] for k,v in pixels.items()},
            image_masks={k:np.asarray([v],bool) for k,v in present.items()},state=state[None],
            tokenized_prompt=tokens[None],tokenized_prompt_mask=mask[None])
        started=time.monotonic()
        result=np.asarray(sample(trainable,frozen,jax.device_put(observation),jax.random.key(seed)))
        if result.shape!=(1,10,7) or not np.isfinite(result).all():raise ValueError('Invalid action tensor')
        raw=denormalize(result[0],action_stats(dataset.normalization,phase))
        return raw,dict(inference_seconds=time.monotonic()-started)
    info=dict(checkpoint=str(checkpoint),checkpoint_step=metadata['step'],
        checkpoint_metadata_sha256=training._sha(checkpoint/'metadata.json'),data_hash=dataset.data_hash,
        sampling_steps=sampling_steps,sampling_seed=seed,
        sampling_policy='same fixed common Gaussian seed at every receding-horizon query',
        modalities=dataset.manifest['modalities'],audio='absent',action_semantics='measured_tcp_command_offset_v4')
    return predict,info
