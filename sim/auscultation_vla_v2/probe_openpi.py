"""Public pi05_base engineering probe. Synthetic inputs, never robot commands.

Run inside the pinned official openpi environment, one process per GPU.
No experiment or patient files are consumed. Result actions are discarded.
"""
import argparse
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import urllib.parse
import urllib.request

import flax.nnx as nnx
from flax import traverse_util
import google_crc32c
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sentencepiece

from openpi.models import model as model_api
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

ROOT = Path('/home/william/wam-vla-v2')


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def public_tokenizer():
    path = ROOT / 'public-assets' / 'paligemma_tokenizer.model'
    meta_url = 'https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model'
    with urllib.request.urlopen(meta_url, timeout=30) as source:
        meta = json.load(source)
    if not path.exists():
        url = 'https://storage.googleapis.com/big_vision/paligemma_tokenizer.model?generation=' + meta['generation']
        with urllib.request.urlopen(url, timeout=60) as source:
            raw = source.read(32 * 2**20)
        path.write_bytes(raw)
        path.chmod(0o600)
    raw = path.read_bytes()
    crc = base64.b64encode(google_crc32c.Checksum(raw).digest()).decode()
    if len(raw) != int(meta['size']) or crc != meta['crc32c']:
        raise ValueError('Public tokenizer integrity failure')
    write(ROOT / 'public_tokenizer_verified.json', dict(source=meta_url, bytes=len(raw),
        crc32c_verified=True, sha256=hashlib.sha256(raw).hexdigest(), generation=meta['generation']))
    # Use the official tokenization method with a separately verified public file.
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=raw)
    return tokenizer


def host_observation(tokenizer):
    state = np.zeros(32, np.float32)  # Synthetic normalized state, not physical features.
    tokens, mask = tokenizer.tokenize('Approach the designated target.', state)
    keys = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
    return model_api.Observation(images={key: np.zeros((1,224,224,3),np.float32) for key in keys},
        image_masks={key: np.array([i == 0]) for i,key in enumerate(keys)},
        state=state[None], tokenized_prompt=tokens.astype(np.int32)[None],
        tokenized_prompt_mask=mask[None])


def synchronize(value):
    jax.block_until_ready(value)
    return np.asarray(value)


def stats(samples):
    return dict(samples_s=samples, median_s=float(np.median(samples)), p95_s=float(np.percentile(samples,95)))


def resource_snapshot():
    return dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip())


def inference(params, tokenizer, horizon, report, output):
    cfg = Pi0Config(pi05=True, action_horizon=horizon)
    model = cfg.load(params)
    model.eval()
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=('num_steps',))
    obs = jax.device_put(host_observation(tokenizer))
    jax.block_until_ready(obs)
    rng = jax.random.key(42)
    for steps in (5,10,20):
        before = time.perf_counter()
        actions = synchronize(sample(rng,obs,num_steps=steps))
        cold = time.perf_counter()-before
        if actions.shape != (1,horizon,32) or not np.isfinite(actions).all():
            raise ValueError('Invalid native sampling output')
        # Two extra warmups after compilation. Fixed RNG makes workloads comparable.
        for _ in range(2):
            synchronize(sample(rng,obs,num_steps=steps))
        gpu_only, host_to_output = [], []
        for _ in range(20):
            before=time.perf_counter()
            synchronize(sample(rng,obs,num_steps=steps))
            gpu_only.append(time.perf_counter()-before)
            before=time.perf_counter()
            fresh = jax.device_put(host_observation(tokenizer))
            synchronize(sample(rng,fresh,num_steps=steps))
            host_to_output.append(time.perf_counter()-before)
        row=dict(horizon=horizon, flow_steps=steps, cold_compile_and_first_sample_s=cold,
            warmups=2, accelerator_input_to_materialized_output=stats(gpu_only),
            synthetic_tokenization_transfer_sampling_output=stats(host_to_output),
            output_shape=list(actions.shape), output_finite=True, resources=resource_snapshot())
        report['inference'].append(row)
        write(output,report)
        print(json.dumps(dict(event='inference_complete',horizon=horizon,steps=steps,
                              median_s=row['synthetic_tokenization_transfer_sampling_output']['median_s'],
                              p95_s=row['synthetic_tokenization_transfer_sampling_output']['p95_s'])),flush=True)
    del sample, model, obs
    jax.clear_caches(); gc.collect()


def lora_step(params, tokenizer):
    cfg=Pi0Config(pi05=True,action_horizon=10,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    skeleton=nnx.eval_shape(cfg.create,jax.random.key(0))
    reference=traverse_util.flatten_dict(nnx.state(skeleton).to_pure_dict())
    loaded=traverse_util.flatten_dict(params)
    merged={}
    missing=[]
    for i,(path,spec) in enumerate(reference.items()):
        if path in loaded:
            if loaded[path].shape != spec.shape:
                raise ValueError('Pretrained shape mismatch: '+str(path))
            merged[path]=loaded[path]  # Frozen BF16 checkpoint, no duplicate full initialization.
        elif 'lora' in '/'.join(path):
            # Official LoRAConfig initializes both factors normal(stddev=.01).
            merged[path]=jax.random.normal(jax.random.fold_in(jax.random.key(123),i),spec.shape,jnp.float32)*.01
            missing.append('/'.join(path))
        else:
            raise ValueError('Unexplained missing pretrained key: '+str(path))
    if not missing:
        raise ValueError('No LoRA factors discovered')
    model=cfg.load(traverse_util.unflatten_dict(merged))
    model.train()
    train_filter=nnx.All(nnx.Param,nnx_utils.PathRegex('.*lora.*'))
    graph,trainable,frozen=nnx.split(model,train_filter,...)
    train_count=sum(x.size for x in jax.tree.leaves(trainable))
    frozen_count=sum(x.size for x in jax.tree.leaves(frozen))
    obs=jax.device_put(host_observation(tokenizer))
    actions=jnp.zeros((1,10,32),jnp.float32)
    def loss_fn(trainable,frozen,obs,actions):
        candidate=nnx.merge(graph,trainable,frozen)
        return candidate.compute_loss(jax.random.key(99),obs,actions,train=True).mean()
    grad_fn=jax.jit(jax.value_and_grad(loss_fn,argnums=0))
    before=time.perf_counter()
    loss,grads=grad_fn(trainable,frozen,obs,actions)
    jax.block_until_ready((loss,grads))
    gradient_s=time.perf_counter()-before
    finite=bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(grads)]))))
    norm=float(optax.global_norm(grads))
    if not np.isfinite(float(loss)) or not finite or not norm>0:
        raise ValueError('Invalid LoRA gradient')
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    updates,_=optimizer.update(grads,optimizer.init(trainable),trainable)
    updated=optax.apply_updates(trainable,updates)
    jax.block_until_ready(updated)
    changed=sum(bool(jax.device_get(jnp.any(a!=b))) for a,b in zip(jax.tree.leaves(trainable),jax.tree.leaves(updated)))
    after_model=nnx.merge(graph,updated,frozen)
    _,after_trainable,after_frozen=nnx.split(after_model,train_filter,...)
    unchanged=all(bool(jax.device_get(jnp.array_equal(a,b))) for a,b in zip(jax.tree.leaves(frozen),jax.tree.leaves(after_frozen)))
    if changed==0 or not unchanged:
        raise ValueError('Parameter update/freeze verification failed')
    return dict(synthetic_only=True, batch_size=1,horizon=10,optimizer='clip_global_norm(1)+Adam(lr=1e-4)',
        train_filter='only paths containing lora; ALL other parameters frozen BF16',
        trainable_parameters=train_count,frozen_parameters=frozen_count,trainable_paths=missing,
        loss=float(loss),gradient_norm=norm,gradients_finite=finite,changed_trainable_leaves=changed,
        all_frozen_leaves_unchanged=unchanged,compile_and_gradient_s=gradient_s,resources=resource_snapshot(),
        task_training=False,checkpoint_saved=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['inference','gradient'],required=True)
    args=parser.parse_args()
    if not (ROOT/'public_checkpoint_verified.json').is_file():
        raise RuntimeError('Verified checkpoint receipt required')
    if not jax.devices() or jax.devices()[0].platform!='gpu':
        raise RuntimeError('GPU required; no silent CPU fallback')
    output=ROOT/'results'/'readiness_v1'/('probe_'+args.mode+'.json')
    report=dict(schema='pi05_synthetic_readiness_v1',mode=args.mode,synthetic_only=True,complete=False,
        task_trained=False,closed_loop_ready=False,openpi_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jax=jax.__version__,
        device=str(jax.devices()[0]),initial_resources=resource_snapshot(),inference=[],
        excluded_latency=['camera acquisition','RGB-D tracking','network','physical actuator response'],
        native_32d_output_is_not_deployment_action=True)
    write(output,report)
    tokenizer=public_tokenizer()
    start=time.perf_counter()
    params=model_api.restore_params(ROOT/'public-assets'/'pi05_base'/'params',dtype=jnp.bfloat16)
    jax.block_until_ready(params)
    report['restore_s']=time.perf_counter()-start
    report['parameter_count']=sum(x.size for x in jax.tree.leaves(params))
    report['pretrained_finite']=all(bool(jax.device_get(jnp.isfinite(x).all())) for x in jax.tree.leaves(params))
    if not report['pretrained_finite']:
        raise ValueError('Nonfinite pretrained weights')
    write(output,report)
    print(json.dumps(dict(event='pretrained_restored',parameters=report['parameter_count'],restore_s=report['restore_s'])),flush=True)
    if args.mode=='inference':
        for horizon in (10,50):
            inference(params,tokenizer,horizon,report,output)
    else:
        report['lora']=lora_step(params,tokenizer)
    report['complete']=True
    write(output,report)
    print(json.dumps(dict(event='probe_complete',mode=args.mode,result=str(output))),flush=True)


if __name__=='__main__':
    main()
