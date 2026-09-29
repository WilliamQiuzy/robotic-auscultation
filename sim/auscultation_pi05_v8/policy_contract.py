"""Explicit V8 B-candidate wire identity; no legacy pilot identity substitution."""
import hashlib
from pathlib import Path
import re

from . import action_encoding as codec
from .train import RECIPE, source_hashes

INFERENCE_MODULES = ('actor.py', 'policy_contract.py', 'policy_client.py',
                     'policy_service.py', 'pair_admission.py')
CONFIG_FIELDS = ('checkpoint_metadata_sha256', 'checkpoint_binding', 'pair_admission_sha256',
                 'pair_mode', 'training_source_sha256', 'inference_source_sha256',
                 'data_hash', 'normalization_sha256', 'dataset_codec_binding',
                 'dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256')
SAMPLING_POLICY = 'same fixed common Gaussian seed at every receding-horizon query'


def need(value, message):
    if not value:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None and value != '0' * 64


def inference_source_paths():
    return {'v8/' + n: Path(__file__).with_name(n) for n in INFERENCE_MODULES}


def inference_source_hashes():
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in inference_source_paths().items()}


def verify_runtime_sources(binding):
    need(binding.get('training_source_sha256') == source_hashes(), 'Actual training sources differ')
    need(binding.get('inference_source_sha256') == inference_source_hashes(), 'Actual inference sources differ')


def verify_policy_binding(binding, expected_metadata_sha256):
    from sim.auscultation_pi05_v6.training_data import MODALITIES
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    need(isinstance(binding, dict) and is_sha(expected_metadata_sha256), 'Explicit V8 binding required')
    expected = dict(schema='wam_horizon_pair_policy_binding_v1', pair_mode=codec.B,
        checkpoint_metadata_sha256=expected_metadata_sha256, training_recipe=RECIPE,
        action_semantics='absolute_joint_command_v1',
        model_action_semantics='future_command_minus_fixed_observation_measured_q',
        sampling_steps=10, sampling_seed=917, sampling_policy=SAMPLING_POLICY,
        modalities=MODALITIES, audio='absent', observation_contract=observation_binding(),
        action_encoding_binding=codec.action_encoding_binding(codec.B))
    for key, value in expected.items():
        need(type(binding.get(key)) is type(value) and binding[key] == value, 'Incompatible V8 policy field: ' + key)
    for key in ('pair_admission_sha256', 'data_hash', 'normalization_sha256',
                'dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256'):
        need(is_sha(binding.get(key)), 'Missing V8 policy hash: ' + key)
    checkpoint = binding.get('checkpoint_binding', {})
    need(checkpoint.get('schema') == 'wam_horizon_pair_checkpoint_binding_v1' and
         checkpoint.get('mode') == codec.B and type(checkpoint.get('local_step')) is int and
         checkpoint['local_step'] > 0 and type(checkpoint.get('optimizer_step')) is int and
         checkpoint['optimizer_step'] == checkpoint['local_step'],
         'Positive fresh B checkpoint required')
    for alias, key in (('checkpoint', 'checkpoint_path'), ('checkpoint_step', 'local_step'),
                       ('checkpoint_optimizer_step', 'optimizer_step')):
        need(type(binding.get(alias)) is type(checkpoint.get(key)) and binding[alias] == checkpoint[key],
             'Checkpoint identity alias differs: ' + alias)
    for policy_key, checkpoint_key in (
        ('checkpoint_metadata_sha256', 'metadata_sha256'), ('data_hash', 'data_hash'),
        ('normalization_sha256', 'normalization_sha256'), ('training_source_sha256', 'source_sha256')):
        need(binding.get(policy_key) == checkpoint.get(checkpoint_key), 'Checkpoint/wire binding differs: ' + policy_key)
    dataset = binding.get('dataset_codec_binding', {})
    need(dataset.get('normalization_sha256') == binding['normalization_sha256'] and
         dataset.get('action_encoding_binding') == binding['action_encoding_binding'] and
         dataset.get('dataset_binding', {}).get('mode') == codec.B and
         dataset['dataset_binding'].get('data_hash') == binding['data_hash'], 'Dataset codec/wire differs')
    for key, expected_keys in (('training_source_sha256', set(source_hashes())),
                               ('inference_source_sha256', set(inference_source_paths()))):
        values = binding.get(key)
        need(isinstance(values, dict) and set(values) == expected_keys and all(is_sha(v) for v in values.values()),
             'Complete explicit source binding required: ' + key)


def verify_config_binding(config, binding):
    verify_policy_binding(binding, config.get('expected_checkpoint_metadata_sha256'))
    for key in CONFIG_FIELDS:
        need(config.get('expected_' + key) == binding.get(key) and config.get('expected_' + key) is not None,
             'Explicit V8 service configuration differs: ' + key)


def verify_against_admission(binding, admission, admission_sha256):
    need(admission.get('candidate_admitted') is True and admission.get('service_eligible') is True,
         'Positive reconstructed pair admission required')
    checkpoint = admission['admitted_checkpoint_binding']
    verify_policy_binding(binding, checkpoint['metadata_sha256'])
    need(binding['pair_admission_sha256'] == admission_sha256 and binding['checkpoint_binding'] == checkpoint and
         binding['dataset_codec_binding'] == admission['admitted_dataset_codec_binding'],
         'Policy differs from exact admitted B checkpoint/codec')


def make_binding(admission, admission_sha256, metadata):
    from sim.auscultation_pi05_v6.training_data import MODALITIES
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    checkpoint = admission['admitted_checkpoint_binding']
    config = metadata['config']
    result = dict(schema='wam_horizon_pair_policy_binding_v1', checkpoint=checkpoint['checkpoint_path'],
        checkpoint_step=checkpoint['local_step'], checkpoint_optimizer_step=checkpoint['optimizer_step'],
        checkpoint_binding=checkpoint, checkpoint_metadata_sha256=checkpoint['metadata_sha256'],
        pair_admission_sha256=admission_sha256, pair_mode=codec.B,
        training_source_sha256=source_hashes(), inference_source_sha256=inference_source_hashes(),
        data_hash=checkpoint['data_hash'], normalization_sha256=checkpoint['normalization_sha256'],
        dataset_codec_binding=admission['admitted_dataset_codec_binding'],
        sampling_steps=10, sampling_seed=917, sampling_policy=SAMPLING_POLICY,
        training_recipe=RECIPE, modalities=MODALITIES, audio='absent',
        action_semantics='absolute_joint_command_v1', model_action_semantics='future_command_minus_fixed_observation_measured_q',
        observation_contract=observation_binding(), action_encoding_binding=codec.action_encoding_binding(codec.B),
        **{key: config[key] for key in ('dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256')})
    verify_against_admission(result, admission, admission_sha256)
    verify_runtime_sources(result)
    return result
