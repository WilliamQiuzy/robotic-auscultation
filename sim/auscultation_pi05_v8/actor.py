"""Actual admitted fresh-pair B model with its own horizon codec."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np


def read_bound_json(path, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Admitted input bytes changed before parsing: ' + str(path))
    return json.loads(raw)


def load_actor(admission_path, admission_sha256, *, sampling_steps=10, seed=917):
    from . import pair_admission as pair, train as training, training_data as data
    from . import action_encoding as codec, policy_contract as contract
    contract.need(sampling_steps == 10 and seed == 917, 'Declared sampler remains seed917/10 steps')
    admitted = pair.verify_pair_admission(admission_path, admission_sha256)
    contract.need(admitted.get('candidate_admitted') is True and admitted.get('service_eligible') is True,
                  'Positive reconstructed B admission required')
    binding = admitted['admitted_checkpoint_binding']
    contract.need(binding['mode'] == codec.B, 'Only locked admitted B can serve')
    checkpoint = Path(binding['checkpoint_path']).resolve(strict=True)
    contract.need(training.checkpoint_binding(checkpoint) == binding, 'Admitted checkpoint changed')
    metadata = read_bound_json(checkpoint / 'metadata.json', binding['metadata_sha256'])
    config = metadata['config']
    config_path = Path(config['training_run_path']) / 'configuration.json'
    saved = read_bound_json(config_path, admitted['artifact_sha256'][str(config_path)])
    contract.need(saved['config'] == config and saved['normalization'] == metadata['normalization'] and
                  saved['data_hash'] == binding['data_hash'], 'Admitted checkpoint/config/data changed')
    dataset, schedule, assets, spec = training.load_inputs(saved['manifest'], codec.B)
    training.verify_configuration(dataset, config, schedule, assets, spec)
    codec_binding = dict(dataset_binding=data.dataset_binding(dataset),
        normalization_sha256=training.schedule_api.digest(dataset.normalization),
        action_encoding_binding=dataset.normalization['action_encoding_binding'])
    contract.need(codec_binding == admitted['admitted_dataset_codec_binding'], 'Admitted dataset codec changed')
    import jax
    from openpi.models import model as model_api
    from sim.auscultation_pi05_v7 import model as implementation
    from sim.auscultation_pi05_v7.training_data import observation, stack_observations
    runtime = implementation.initialize(61, 5e-5)
    contract.need(runtime['assets'] == config['assets'], 'Official public assets changed')
    implementation.restore(runtime, checkpoint, dataset.data_hash, config, dataset.normalization)
    contract.need(training.checkpoint_binding(checkpoint) == binding, 'Admitted checkpoint changed during restore')
    sample = implementation.sampler(runtime, steps=10)
    info = contract.make_binding(admitted, admission_sha256, metadata)

    def predict(rgb, state, phase, target):
        raw = np.asarray(state, np.float32)
        contract.need(raw.shape == (41,) and np.isfinite(raw).all(), 'Finite original 41D state required')
        row = observation(rgb, raw, phase, target, runtime['tokenizer'], dataset.normalization)
        obs = model_api.Observation(**jax.device_put(stack_observations([row])))
        started = time.monotonic()
        normalized = np.asarray(sample(runtime['trainable'], runtime['frozen'], obs, jax.random.key(917)))
        contract.need(normalized.shape == (1, 10, 7) and np.isfinite(normalized).all(), 'Malformed residual output')
        absolute = data.decode_prediction(dataset, normalized[0], raw, phase)
        contract.need(absolute.shape == (10, 7) and np.isfinite(absolute).all(), 'Malformed absolute output')
        return absolute, dict(inference_seconds=time.monotonic() - started,
            observation_anchor_q=raw[:7].tolist(), model_normalized_residual=normalized[0].tolist(),
            decoder='V8 B horizon codec then fixed measured q added once; no clipping')

    return predict, info
