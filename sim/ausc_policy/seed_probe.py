"""M0: sample an existing Molmo checkpoint; never train or drive a controller.

The legacy adapter is read-only because this experiment reinterprets TRAIN32.
New action contracts belong to subsequent experiments, not this diagnostic.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_config(config):
    indices, seeds = config['sample_indices'], config['seeds']
    if len(indices) != 20 or len(set(indices)) != 20 or not all(0 <= i < 32 for i in indices):
        raise ValueError('Require 20 distinct TRAIN32 observations')
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise ValueError('Require 16 distinct predetermined seeds')
    if config['checkpoint_step'] != 512 or config['ode_steps'] != 10:
        raise ValueError('M0 reinterprets fixed final512 checkpoint with ODE10')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Requires the Linux GPU owner entrypoint')
    config = json.loads(args.config.read_text())
    validate_config(config)
    base = Path(config['base'])
    with (base / 'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, config, base)


def run(args, config, base):
    started = time.time()
    diag = base / 'diagnostics'
    legacy = diag / 'molmoact2_train32_candidate_v1'
    # These sibling imports are the existing checkpoint format, not a new copy.
    sys.path.insert(0, str(legacy))
    trainer = load_module('_m0_legacy_train32', legacy / 'train.py')
    ckpt = trainer.ckpt
    adapter, probe = trainer.load_dependencies(diag / 'molmoact2_wam_probe_candidate_v1')
    pins_path = diag / 'molmoact2_wam_probe_launch_v1/pins.json'
    pins = probe.verify_pins(pins_path, config['pins_sha256'])
    probe.gpu_admission()
    trained = diag / 'molmoact2_train32_actual_v1'
    release = json.loads((diag / 'molmoact2_train32_execution_v1/lifecycle_result.json').read_text())
    if not (release['completed'] and release['gpu_released'] and release['stage']['process_group_drained']):
        raise RuntimeError('Previous TRAIN32 owner has not completed and released GPU')
    context = json.loads((trained / 'configuration.json').read_text())
    directory = trained / 'checkpoint_0512'
    if ckpt.sha(directory / 'COMPLETE.json') != config['checkpoint_complete_sha256']:
        raise ValueError('Predetermined checkpoint identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    ckpt.save_json(args.output / 'configuration.json', config)
    data = adapter.load_prepared(diag / 'pi05_v12_train_fit_prepared_v1', pins['runtime'])
    cfg, packer = probe.native(pins, 'cuda')
    if ckpt.canonical(dataclasses.asdict(cfg)) != context['config']:
        raise ValueError('Native checkpoint configuration differs')
    packed = []
    for index in config['sample_indices']:
        batch, receipt = adapter.pack_row(data, data.row(index), packer, include_action=False)
        if 'action' in batch or 'labels' in batch:
            raise ValueError('Prediction input contains labels')
        packed.append(batch)
    del packer
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    ckpt.seed_all(torch, 61)
    policy = MolmoAct2Policy(cfg).to('cuda')
    restore = ckpt.load_trainable_parameters(directory, policy, torch,
        config['checkpoint_complete_sha256'], 512, context)
    ckpt.save_json(args.output / 'checkpoint_loaded.json', dict(cursor=restore['cursor'],
        trainable_tensors=len(restore['restored_parameters']),
        optimizer_loaded=False, checkpoint_complete_sha256=config['checkpoint_complete_sha256']))
    probe.reset_modulation_cache(policy._action_expert())
    raw = np.empty((20, 16, 10, 7), dtype=np.float32)
    normalized = np.empty_like(raw)
    prediction_seconds = []
    with ckpt.isolated_evaluation(policy, torch):
        for i, (index, batch) in enumerate(zip(config['sample_indices'], packed)):
            on_device = trainer.move_batch(batch, torch)
            for j, seed in enumerate(config['seeds']):
                ckpt.seed_all(torch, seed)
                before = time.monotonic()
                with torch.inference_mode():
                    prediction = policy.predict_action_chunk(on_device,
                        generator=torch.Generator(device='cuda').manual_seed(seed), num_steps=10)
                value = prediction[0].detach().cpu().numpy().copy()
                prediction_seconds.append(time.monotonic() - before)
                if value.shape != (10, 7) or not np.isfinite(value).all():
                    raise ValueError('Invalid prediction')
                normalized[i, j] = value
                raw[i, j] = data.decode(value, data.row(index))
            ckpt.save_json(args.output / 'progress.json', dict(observations_done=i+1,
                predictions_done=(i+1)*16, total_predictions=320))
    ix = config['sample_indices']
    np.savez(args.output / 'seed_predictions.npz', raw=raw,
        normalized=normalized, reference=data.arrays['reference'][ix],
        measured=data.arrays['state'][ix, :7], sample_indices=np.asarray(ix),
        seeds=np.asarray(config['seeds']), prediction_seconds=np.asarray(prediction_seconds))
    ckpt.save_json(args.output / 'COMPLETE.json', dict(completed=True, predictions=320,
        wall_seconds=time.time()-started, checkpoint_step=512, optimizer_updates=0,
        array_sha256=ckpt.sha(args.output / 'seed_predictions.npz'),
        scope='TRAIN32 sampling dispersion only; no controller or generalization claim'))
    print(json.dumps(dict(completed=True, predictions=320)), flush=True)


if __name__ == '__main__':
    main()
