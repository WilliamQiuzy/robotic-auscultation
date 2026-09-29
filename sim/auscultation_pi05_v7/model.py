"""Unmodified pinned official pi05 flow head and V6 LoRA partition.

This first ablation changes action representation alone, not architecture,
tokenization, optimizer, trainable parameter selection or physics.
"""
import inspect
from pathlib import Path
import sys
import numpy as np
from sim.auscultation_pi05_v6 import train_openpi as parent
from sim.auscultation_pi05_v6.openpi_adapter import trainable_path

def initialize(seed,learning_rate):
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    import sentencepiece
    from openpi.models import model as model_api
    from openpi.models.pi0_config import Pi0Config
    repo=parent.ROOT/'vendor/openpi'
    if Path(sys.prefix).resolve()!=(repo/'.venv').resolve():raise RuntimeError('Official OpenPI Python required')
    parent.verify_openpi_checkout(repo,inspect.getfile(Pi0Config))
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise RuntimeError('Exactly one GPU required')
    params_path,tokenizer_path,assets=parent._verify_public_assets(parent.ROOT)
    tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    cfg=Pi0Config(pi05=True,action_dim=7,action_horizon=10,max_token_len=256,
                 paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    skeleton=nnx.eval_shape(cfg.create,jax.random.key(seed))
    reference=traverse_util.flatten_dict(nnx.state(skeleton).to_pure_dict())
    pretrained=traverse_util.flatten_dict(model_api.restore_params(params_path,dtype=jnp.bfloat16))
    path_ids={path:i for i,path in enumerate(reference)}
    merged=parent.merge_pretrained(reference,pretrained,
        lambda path,shape:jax.random.normal(jax.random.fold_in(jax.random.key(seed),path_ids[path]),shape,jnp.float32)*.01)
    candidate=cfg.load(traverse_util.unflatten_dict(merged));candidate.train()
    graph,trainable,frozen=nnx.split(candidate,nnx.All(nnx.Param,lambda path,value:trainable_path(path)),...)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(learning_rate))
    opt_state=optimizer.init(trainable)
    leaves,definition=jax.tree_util.tree_flatten_with_path((trainable,opt_state))
    contract=[dict(path=jax.tree_util.keystr(path),shape=list(value.shape),dtype=str(value.dtype)) for path,value in leaves]
    return dict(graph=graph,trainable=trainable,frozen=frozen,optimizer=optimizer,opt_state=opt_state,
                tree_definition=definition,leaf_contract=contract,assets=assets,tokenizer=tokenizer)

def restore(runtime,checkpoint,data_hash,config,normalization):
    import jax
    import jax.numpy as jnp
    arrays,metadata=parent.load_checkpoint(checkpoint,data_hash=data_hash,config=config,normalization=normalization)
    if metadata['leaf_contract']!=runtime['leaf_contract']:raise ValueError('Checkpoint tree contract differs')
    expected={'leaf_%05d'%i for i in range(len(runtime['leaf_contract']))}|{'jax_rng'}
    if set(arrays)!=expected:raise ValueError('Checkpoint arrays differ')
    leaves=[]
    for i,contract in enumerate(runtime['leaf_contract']):
        value=arrays['leaf_%05d'%i]
        if list(value.shape)!=contract['shape'] or str(value.dtype)!=contract['dtype']:
            raise ValueError('Checkpoint leaf shape/dtype differs')
        leaves.append(jnp.asarray(value))
    runtime['trainable'],runtime['opt_state']=jax.tree_util.tree_unflatten(runtime['tree_definition'],leaves)
    return metadata,arrays['jax_rng']

def sampler(runtime,steps=10):
    import flax.nnx as nnx
    import jax
    @jax.jit
    def sample(parameters,frozen,observation,rng):
        model=nnx.merge(runtime['graph'],parameters,frozen);model.eval()
        return model.sample_actions(rng,observation,num_steps=steps)
    return sample
