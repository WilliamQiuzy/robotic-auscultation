"""Portable ACT reference for qualified Isaac arrays, never a robot controller.

Matches the released state17/history2/H20/E10 numerical contract. This small
entry point is separate from the original owned experiment launcher; running
it does not recreate the original run receipts or confer physical acceptance.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

ARRAY_SHAPES = {'states': (17,), 'histories': (2, 17), 'actions': (20, 7),
                'origins': (7,), 'velocities': (7,), 'observation_times': ()}
MODEL_SPEC = dict(n_obs_steps=1, chunk_size=20, n_action_steps=10,
                  temporal_ensemble_coeff=None, dim_model=256, n_heads=4,
                  n_encoder_layers=2, n_decoder_layers=1, dim_feedforward=1024,
                  use_vae=False, dropout=.1, pretrained_backbone_weights=None,
                  vision_backbone='resnet18')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_split_ids(train, validation):
    need(bool(train) and bool(validation), 'Nonempty identity lists required')
    need(all(isinstance(x, str) and x for x in [*train, *validation]), 'Invalid subject identifier')
    need(len(set(train)) == len(train) and len(set(validation)) == len(validation), 'Duplicate subject identifier')
    need(not set(train) & set(validation), 'TRAIN / validation identity overlap')


def validate_arrays(samples):
    need(set(samples) == set(ARRAY_SHAPES), 'Exactly six numerical arrays required')
    n = len(samples['states'])
    need(n > 0, 'Empty dataset')
    for key, tail in ARRAY_SHAPES.items():
        a = np.asarray(samples[key])
        need(a.shape == (n,) + tail and a.dtype.kind in 'fiu' and np.isfinite(a).all(),
             'Invalid array: ' + key)
    need(np.array_equal(samples['histories'][:, 1], samples['states']), 'History/current state mismatch')
    need(np.array_equal(samples['origins'], samples['states'][:, :7]), 'Block origin/current joint mismatch')
    expected = (samples['histories'][:, 1, :7] - samples['histories'][:, 0, :7]) / .05
    need(np.allclose(samples['velocities'], expected, atol=1e-9, rtol=1e-9), 'Causal velocity mismatch')


def load_dataset(path):
    path = Path(path).resolve(); manifest = json.loads(path.read_text())
    need(manifest.get('schema') == 'ausc-act-arrays-v1', 'Use the documented array manifest')
    need(manifest.get('action_semantics') == 'raw_joint_target_minus_block_origin', 'Wrong action representation')
    need(set(manifest.get('splits', {})) == {'train', 'validation'}, 'Only TRAIN and validation are accepted')
    rows = manifest['splits']; validate_split_ids(rows['train']['subject_ids'], rows['validation']['subject_ids'])
    result = {}
    for split, meta in rows.items():
        array_path = (path.parent / meta['file']).resolve()
        need(sha256(array_path) == meta['sha256'], 'Array checksum mismatch: ' + split)
        with np.load(array_path, allow_pickle=False) as packet:
            samples = {k: packet[k] for k in packet.files}
        validate_arrays(samples)
        labels = meta.get('window_subject_ids', [])
        need(len(labels) == len(samples['states']) and set(labels) == set(meta['subject_ids']),
             'Complete per-window subject mapping required')
        result[split] = samples
    return result, manifest


def normalization(train):
    validate_arrays(train)
    return dict(fitted_split='train',
                state_mean=train['histories'].mean((0, 1)).tolist(),
                state_std=np.maximum(train['histories'].std((0, 1)), 1e-6).tolist(),
                action_mean=train['actions'].mean((0, 1)).tolist(),
                action_std=np.maximum(train['actions'].std((0, 1)), 1e-6).tolist())


def pack(samples, stats, ids):
    need(stats.get('fitted_split') == 'train', 'Normalization must be TRAIN-only')
    h = (samples['histories'][ids] - np.array(stats['state_mean'])) / np.array(stats['state_std'])
    actions = (samples['actions'][ids] - np.array(stats['action_mean'])) / np.array(stats['action_std'])
    return {'observation.state': h[:, 1, :7].astype(np.float32),
            'observation.environment_state': np.concatenate([h[:, 1, 7:], h[:, 0]], axis=1).astype(np.float32),
            'action': actions.astype(np.float32),
            'action_is_pad': np.zeros((len(h), 20), dtype=bool)}


def absolute_chunk(relative, origin):
    delta = np.asarray(relative); q = np.asarray(origin)
    need(delta.shape == (20, 7) and q.shape == (7,) and np.isfinite(delta).all() and np.isfinite(q).all(),
         'Finite H20 delta and one fixed seven-joint origin required')
    return delta + q[None, :]


def make_model(device):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = ACTConfig(
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (7,)),
                        'observation.environment_state': PolicyFeature(FeatureType.ENV, (27,))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))},
        normalization_mapping={k: NormalizationMode.IDENTITY for k in ('STATE', 'ENV', 'ACTION')},
        **MODEL_SPEC, device=device, use_amp=False, push_to_hub=False)
    policy = ACTPolicy(config).to(device)
    need(sum(p.numel() for p in policy.parameters()) == 2658823, 'Official ACT parameter count changed')
    return torch, policy


def evaluate(torch, policy, data, stats, device):
    policy.eval(); predictions = []
    with torch.inference_mode():
        for start in range(0, len(data['states']), 256):
            ids = np.arange(start, min(start + 256, len(data['states'])))
            batch = {k: torch.from_numpy(v).to(device) for k, v in pack(data, stats, ids).items()}
            value = policy.predict_action_chunk(batch).cpu().numpy()
            predictions.append(value * np.array(stats['action_std']) + np.array(stats['action_mean']))
    prediction = np.concatenate(predictions)
    truth = data['actions']
    velocity = data['velocities'][:, None, :] * (np.arange(1, 21) * .05)[None, :, None]
    return {name: float(np.abs(value - truth).mean()) for name, value in
            [('act_mae_rad', prediction), ('hold_mae_rad', np.zeros_like(truth)), ('velocity_mae_rad', velocity)]}


@contextmanager
def gpu_lease(path):
    import fcntl
    need(bool(path), 'An explicit GPU coordination lock is required for CUDA')
    with Path(path).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def fit(args):
    from contextlib import nullcontext
    need(args.steps > 0 and 0 < args.eval_every <= args.steps and 0 < args.batch_size <= 256,
         'Positive bounded schedule required')
    data, manifest = load_dataset(args.manifest)
    stats = normalization(data['train'])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    lease = gpu_lease(args.gpu_lock) if args.device == 'cuda' else nullcontext()
    started = time.monotonic()
    with lease:
        import torch
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.device == 'cuda':
            need(torch.cuda.is_available(), 'CUDA unavailable; no CPU substitution')
            torch.cuda.manual_seed_all(args.seed)
        torch, policy = make_model(args.device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed); best = float('inf'); best_step = None
        with (output / 'metrics.jsonl').open('x') as log:
            for step in range(1, args.steps + 1):
                policy.train(); ids = rng.integers(len(data['train']['states']), size=args.batch_size)
                batch = {k: torch.from_numpy(v).to(args.device) for k, v in pack(data['train'], stats, ids).items()}
                optimizer.zero_grad(set_to_none=True); loss, _ = policy(batch)
                need(bool(torch.isfinite(loss)), 'Nonfinite ACT loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step % args.eval_every and step != args.steps:
                    continue
                row = dict(step=step, elapsed_wall_s=time.monotonic() - started,
                           **{split: evaluate(torch, policy, values, stats, args.device) for split, values in data.items()})
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
                if row['validation']['act_mae_rad'] < best:
                    best = row['validation']['act_mae_rad']; best_step = step
                    torch.save(dict(state_dict=policy.state_dict(), model_spec=MODEL_SPEC,
                                    normalization=stats, step=step, dataset_manifest_sha256=sha256(args.manifest),
                                    split_ids={s: m['subject_ids'] for s, m in manifest['splits'].items()},
                                    state_only=True, closed_loop_evaluated=False), output / 'best.pt')
        receipt = dict(schema='ausc-portable-act-training-v1', completed_steps=args.steps,
                       selected_step=best_step, selected_validation_mae_rad=best,
                       elapsed_wall_s=time.monotonic() - started, model_spec=MODEL_SPEC,
                       manifest_sha256=sha256(args.manifest), checkpoint_sha256=sha256(output / 'best.pt'),
                       state_only=True, closed_loop_evaluated=False,
                       original_experiment_reproduction=False)
        (output / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=5000); parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64); parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); parser.add_argument('--gpu-lock')
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
