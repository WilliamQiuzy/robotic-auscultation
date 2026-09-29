"""Synthetic engineering probe for the pinned official pi0.5 task adapter.

This is not task training: no Dataset, qualification, train(), robot, or sensor
is used. The numeric snapshot exists only to exercise serialization helpers.
Run in the existing official openpi virtual environment on an otherwise idle GPU.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from .openpi_adapter import (LORA_PATHS, OPENPI_COMMIT, TASK_PROJECTION_PATHS,
                             prepare_images, state_prompt, tokenize_state, trainable_path)
from .train_openpi import (_atomic_json, _sha, _verify_public_assets, load_checkpoint,
                           merge_pretrained, save_checkpoint, training_instruction,
                           verify_openpi_checkout)


def gpu_snapshot():
    return dict(gpu=subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip(),
        compute_processes=subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True).strip(),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run(root, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new empty engineering probe output directory')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='wam_pi05_task_adapter_engineering_probe_v1', complete=False,
                  synthetic_only=True, task_training=False, task_trained=False,
                  saved_task_checkpoint=None, closed_loop_validated=False,
                  script_sha256=_sha(__file__), stages=[], initial_resources=gpu_snapshot())

    def record(stage, **fields):
        report.update(fields)
        report['stages'].append(dict(stage=stage, elapsed_s=time.perf_counter() - started))
        _atomic_json(output / 'report.json', report)
        print(json.dumps(dict(event=stage, elapsed_s=report['stages'][-1]['elapsed_s']),
                         allow_nan=False), flush=True)

    started = time.perf_counter()
    try:
        if report['initial_resources']['compute_processes']:
            raise RuntimeError('GPU compute jobs already exist; no concurrent probe allowed')
        repo = root / 'vendor' / 'openpi'
        if Path(sys.prefix).resolve() != (repo / '.venv').resolve():
            raise RuntimeError('existing official openpi virtual environment required')
        import flax
        import flax.nnx as nnx
        from flax import traverse_util
        import jax
        import jax.numpy as jnp
        import optax
        import sentencepiece
        from openpi.models import model as model_api
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer

        commit = verify_openpi_checkout(repo, inspect.getfile(Pi0Config))
        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise RuntimeError('one GPU required; no CPU fallback')
        params_path, tokenizer_path, assets = _verify_public_assets(root)
        source_files = ['probe_task_adapter.py', 'train_openpi.py', 'openpi_adapter.py',
                        'training_data.py', 'multimodal.py', 'chunks.py', 'hall.py',
                        'observations.py', '__init__.py']
        config = dict(openpi_commit=commit, action_dim=7, state_width=32, horizon=10,
                      max_token_len=200, paligemma_variant='gemma_2b_lora',
                      action_expert_variant='gemma_300m_lora', assets=assets,
                      jax=jax.__version__, flax=flax.__version__, optax=optax.__version__,
                      seed=42, optimizer='clip_global_norm(1)+Adam(lr=1e-4)',
                      task_training=False, saved_task_checkpoint=None)
        record('environment_verified', config=config,
               source_sha256={name: _sha(Path(__file__).with_name(name)) for name in source_files},
               official_source=str(inspect.getfile(Pi0Config)), device=str(jax.devices()[0]))

        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        state = np.linspace(-.75, .75, 32, dtype=np.float32)
        instruction = training_instruction('APPROACH', 'engineering-probe')
        token, mask = tokenize_state(tokenizer, instruction, state)
        official = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
        official._tokenizer, official._max_len = tokenizer, 200
        official_token, official_mask = official.tokenize(instruction, state)
        if not np.array_equal(token, official_token) or not np.array_equal(mask, official_mask):
            raise ValueError('adapter tokenization differs from official bounded-state tokenization')
        prompt = state_prompt(instruction, state)
        discrete = prompt.split(', State: ', 1)[1].split(';', 1)[0].split()
        if len(discrete) != 32:
            raise ValueError('not all state channels serialized into the prompt')
        channel_changes = []
        for channel in range(32):
            changed_state = state.copy()
            changed_state[channel] += -.25 if state[channel] > .5 else .25
            changed_token, _ = tokenize_state(tokenizer, instruction, changed_state)
            channel_changes.append(not np.array_equal(token, changed_token))
        if not all(channel_changes):
            raise ValueError('a state channel does not affect tokenized prompt')

        def observation(input_state, variant):
            # Patterned synthetic RGB has no acquisition, anatomy, or qualification claim.
            yy, xx = np.indices((180, 240))
            rgb = np.stack([(xx + variant * 7) % 256, (yy * 2) % 256,
                            (xx + yy + variant * 11) % 256], axis=-1).astype(np.uint8)
            images, image_masks = prepare_images(rgb)
            ids, valid = tokenize_state(tokenizer, instruction, input_state)
            return model_api.Observation(
                images={key: value[None] for key, value in images.items()},
                image_masks={key: np.array([value]) for key, value in image_masks.items()},
                state=input_state[None], tokenized_prompt=ids[None], tokenized_prompt_mask=valid[None])

        obs = jax.device_put(observation(state, 0))
        validation_state = -state
        val_obs = jax.device_put(observation(validation_state, 1))
        perturb_state = validation_state.copy()
        perturb_state[31] += .25
        perturb_obs = jax.device_put(observation(perturb_state, 1))
        actions = jnp.asarray(np.linspace(-.1, .1, 70, dtype=np.float32).reshape(1, 10, 7))
        val_actions = -actions
        record('synthetic_inputs_verified', inputs=dict(
            state_shape=list(obs.state.shape), action_shape=list(actions.shape),
            token_shape=list(obs.tokenized_prompt.shape), active_tokens=int(mask.sum()),
            state_channels_in_prompt=len(discrete), every_state_channel_changes_tokens=True,
            official_bounded_state_tokenization_exact=True, normalized_synthetic_values=True,
            validation_is_synthetic_engineering_only=True,
            pi05_state_route='all 32 channels discretized into language prefix; raw state is not a continuous state projection'))

        cfg = Pi0Config(pi05=True, action_dim=7, action_horizon=10, max_token_len=200,
                        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
        skeleton = nnx.eval_shape(cfg.create, jax.random.key(42))
        reference = traverse_util.flatten_dict(nnx.state(skeleton).to_pure_dict())
        pretrained = traverse_util.flatten_dict(model_api.restore_params(params_path, dtype=jnp.bfloat16))
        jax.block_until_ready(pretrained)
        if not all(bool(jax.device_get(jnp.isfinite(value).all())) for value in pretrained.values()):
            raise ValueError('nonfinite pretrained parameter')
        count = sum(value.size for value in pretrained.values())
        if count != 3353433872:
            raise ValueError('unexpected public base parameter count')
        indices = {path: i for i, path in enumerate(reference)}
        merged = merge_pretrained(reference, pretrained,
            lambda path, shape: jax.random.normal(jax.random.fold_in(jax.random.key(42), indices[path]),
                                                  shape, jnp.float32) * .01)
        projections = {}
        for path in sorted(TASK_PROJECTION_PATHS):
            original, current = pretrained[path], merged[path]
            expected = (original[:7, :] if path == ('action_in_proj', 'kernel') else
                        original[:, :7] if path == ('action_out_proj', 'kernel') else
                        original[:7] if path == ('action_out_proj', 'bias') else original)
            if not bool(jax.device_get(jnp.array_equal(current, expected.astype(jnp.float32)))):
                raise ValueError('task projection is not the declared exact pretrained slice')
            projections['/'.join(path)] = dict(native_shape=list(original.shape),
                                               task_shape=list(current.shape), exact_pretrained_slice=True)
        actual_lora = {path for path in reference if any('lora' in str(part) for part in path)}
        if actual_lora != LORA_PATHS or len(actual_lora) != 20:
            raise ValueError('actual official LoRA paths differ from 20-path allowlist')
        model = cfg.load(traverse_util.unflatten_dict(merged))
        model.train()
        train_filter = nnx.All(nnx.Param, lambda path, value: trainable_path(path))
        graph, trainable, frozen = nnx.split(model, train_filter, ...)
        selected = set(traverse_util.flatten_dict(trainable.to_pure_dict()))
        if selected != LORA_PATHS | TASK_PROJECTION_PATHS:
            raise ValueError('optimizer partition differs from exact LoRA plus projection allowlist')
        record('model_partition_verified', pretrained_parameter_count=count, projections=projections,
               partition=dict(lora_leaves=len(actual_lora), projection_leaves=len(TASK_PROJECTION_PATHS),
                              trainable_paths=['/'.join(path) for path in sorted(selected)],
                              trainable_parameters=sum(value.size for value in jax.tree.leaves(trainable)),
                              frozen_parameters=sum(value.size for value in jax.tree.leaves(frozen)),
                              trainable_dtypes=sorted({str(value.dtype) for value in jax.tree.leaves(trainable)}),
                              frozen_dtypes=sorted({str(value.dtype) for value in jax.tree.leaves(frozen)})))
        del skeleton, pretrained, merged, model

        trace_shapes = {}
        def objective(parameters, frozen_parameters, observation, target, rng, training):
            if observation.state.shape != (1, 32) or target.shape != (1, 10, 7):
                raise ValueError('actual traced observation/action dimensions are wrong')
            trace_shapes['gradient' if training else 'evaluation'] = dict(
                state=list(observation.state.shape), tokenized_prompt=list(observation.tokenized_prompt.shape),
                target=list(target.shape))
            candidate = nnx.merge(graph, parameters, frozen_parameters)
            candidate.train() if training else candidate.eval()
            return candidate.compute_loss(rng, observation, target, train=training).mean()

        gradient = jax.jit(jax.value_and_grad(lambda p, f, o, a, r: objective(p, f, o, a, r, True), argnums=0))
        evaluate = jax.jit(lambda p, f, o, a, r: objective(p, f, o, a, r, False))
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-4))
        opt_state = optimizer.init(trainable)
        @jax.jit
        def optimize(parameters, state, grads):
            updates, next_state = optimizer.update(grads, state, parameters)
            return optax.apply_updates(parameters, updates), next_state

        rng, loss_rng = jax.random.split(jax.random.key(42))
        before = time.perf_counter()
        loss, grads = gradient(trainable, frozen, obs, actions, loss_rng)
        jax.block_until_ready((loss, grads))
        gradient_s = time.perf_counter() - before
        gradient_norm = float(optax.global_norm(grads))
        if not np.isfinite(float(loss)) or not np.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError('invalid actual loss or gradient norm')
        if not all(bool(jax.device_get(jnp.isfinite(value).all())) for value in jax.tree.leaves(grads)):
            raise ValueError('nonfinite actual gradient')
        updated, updated_opt = optimize(trainable, opt_state, grads)
        jax.block_until_ready((updated, updated_opt))
        before_leaves = traverse_util.flatten_dict(trainable.to_pure_dict())
        after_leaves = traverse_util.flatten_dict(updated.to_pure_dict())
        changed = [path for path in selected if bool(jax.device_get(jnp.any(before_leaves[path] != after_leaves[path])))]
        if set(changed) != selected:
            raise ValueError('not every selected trainable leaf changed')
        after_model = nnx.merge(graph, updated, frozen)
        _, _, after_frozen = nnx.split(after_model, train_filter, ...)
        unchanged = all(bool(jax.device_get(jnp.array_equal(a, b)))
                        for a, b in zip(jax.tree.leaves(frozen), jax.tree.leaves(after_frozen)))
        if not unchanged:
            raise ValueError('frozen parameters changed')
        record('one_engineering_optimizer_update_completed', update=dict(
            optimizer_updates=1, loss=float(loss), gradient_norm=gradient_norm, gradients_finite=True,
            changed_lora_leaves=sum(path in LORA_PATHS for path in changed),
            changed_projection_leaves=sum(path in TASK_PROJECTION_PATHS for path in changed),
            all_frozen_leaves_unchanged=unchanged, compile_and_gradient_s=gradient_s,
            task_training=False), resources_after_update=gpu_snapshot())
        del trainable, opt_state, before_leaves, after_leaves, after_model, after_frozen

        val_rng = jax.random.key(100)
        val_loss = float(evaluate(updated, frozen, val_obs, val_actions, val_rng))
        perturbed_loss = float(evaluate(updated, frozen, perturb_obs, val_actions, val_rng))
        if not np.isfinite([val_loss, perturbed_loss]).all() or val_loss == perturbed_loss:
            raise ValueError('finite evaluation and 32nd-channel effect were not demonstrated')
        @jax.jit
        def sample(parameters, frozen_parameters, observation, rng):
            if observation.state.shape != (1, 32):
                raise ValueError('sampling trace lost state width')
            trace_shapes['sampling'] = dict(state=list(observation.state.shape),
                                            tokenized_prompt=list(observation.tokenized_prompt.shape))
            candidate = nnx.merge(graph, parameters, frozen_parameters)
            candidate.eval()
            return candidate.sample_actions(rng, observation, num_steps=5)
        before = time.perf_counter()
        sampled = np.asarray(sample(updated, frozen, val_obs, jax.random.key(101)))
        sample_s = time.perf_counter() - before
        if sampled.shape != (1, 10, 7) or not np.isfinite(sampled).all():
            raise ValueError('invalid adapted actual sample')
        record('evaluation_and_sampling_verified', evaluation=dict(
            synthetic_loss=val_loss, channel_32_perturbed_loss=perturbed_loss,
            only_channel_32_and_its_tokens_changed=True, channel_32_changes_actual_loss=True,
            sample_shape=list(sampled.shape), sample_finite=True, flow_steps=5,
            compile_and_first_sample_s=sample_s, task_validation=False), traced_shapes=trace_shapes)

        sampler = np.random.default_rng(42)
        sampler.integers(0, 100, size=3)
        sampler_state = sampler.bit_generator.state
        tree = (updated, updated_opt)
        path_leaves, definition = jax.tree_util.tree_flatten_with_path(tree)
        leaf_contract = [dict(path=jax.tree_util.keystr(path), shape=list(value.shape), dtype=str(value.dtype))
                         for path, value in path_leaves]
        arrays = {'leaf_%05d' % i: np.asarray(value) for i, (_, value) in enumerate(path_leaves)}
        arrays['jax_rng'] = np.asarray(jax.random.key_data(rng))
        input_hash = hashlib.sha256(state.tobytes() + np.asarray(actions).tobytes() + token.tobytes()).hexdigest()
        metadata = dict(data_hash=input_hash, config=config, normalization=None,
                        data_kind='synthetic_engineering_probe_not_task_data', task_training=False,
                        saved_task_checkpoint=None, sampler_rng_state=sampler_state, leaf_contract=leaf_contract)
        snapshot = save_checkpoint(output / 'engineering_serialization_only', step=1,
                                   arrays=arrays, metadata=metadata)
        restored_arrays, restored_meta = load_checkpoint(snapshot, data_hash=input_hash,
                                                         config=config, normalization=None)
        if set(arrays) != set(restored_arrays) or restored_meta['leaf_contract'] != leaf_contract:
            raise ValueError('serialization leaf contract mismatch')
        if not all(np.array_equal(value, restored_arrays[key]) for key, value in arrays.items()):
            raise ValueError('serialization changed a numeric leaf')
        restored, restored_opt = jax.tree_util.tree_unflatten(definition,
            [jnp.asarray(restored_arrays['leaf_%05d' % i]) for i in range(len(path_leaves))])
        restored_rng = jax.random.wrap_key_data(jnp.asarray(restored_arrays['jax_rng']))
        restored_sampler = np.random.default_rng()
        restored_sampler.bit_generator.state = restored_meta['sampler_rng_state']
        if not np.array_equal(sampler.integers(0, 100, size=8), restored_sampler.integers(0, 100, size=8)):
            raise ValueError('sampler RNG numerical continuation mismatch')
        if not np.array_equal(jax.random.key_data(jax.random.split(rng)),
                              jax.random.key_data(jax.random.split(restored_rng))):
            raise ValueError('JAX RNG continuation mismatch')
        restored_loss = float(evaluate(restored, frozen, val_obs, val_actions, val_rng))
        restored_sample = np.asarray(sample(restored, frozen, val_obs, jax.random.key(101)))
        if restored_loss != val_loss or not np.array_equal(sampled, restored_sample):
            raise ValueError('restored model numerical outputs differ')
        if not all(bool(jax.device_get(jnp.array_equal(a, b)))
                   for a, b in zip(jax.tree.leaves(updated_opt), jax.tree.leaves(restored_opt))):
            raise ValueError('optimizer state restoration differs')
        record('numerical_serialization_roundtrip_verified', serialization=dict(
            engineering_snapshot_directory=str(snapshot), saved_task_checkpoint=None,
            task_training=False, arrays_sha256=restored_meta['arrays_sha256'],
            numeric_leaf_count=len(arrays), trainable_and_complete_optimizer_exact=True,
            jax_rng_continuation_exact=True, sampler_rng_continuation_exact=True,
            validation_loss_exact=True, sample_output_exact=True,
            serialization_bytes=(snapshot / 'arrays.npz').stat().st_size))
        verify_openpi_checkout(repo, inspect.getfile(Pi0Config))
        record('probe_complete', complete=True, final_resources=gpu_snapshot(), elapsed_s=time.perf_counter() - started)
        return report
    except Exception as error:
        record('probe_failed', error_type=type(error).__name__, error=str(error), complete=False)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/william/wam-vla-v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
