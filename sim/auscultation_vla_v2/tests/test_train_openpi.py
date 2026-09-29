"""Synthetic tmp_path fixtures test gates, never constitute qualified training data."""
import importlib
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.auscultation_vla_v2.training_data import MODALITIES


def api():
    try:
        module = importlib.import_module('sim.auscultation_vla_v2.train_openpi')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'sealed-dataset pi05 training entrypoint is missing'
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


CHECKS = ('rgbd_tracking_valid', 'measured_state_valid', 'hall_valid', 'synchronized',
          'actions_bounded', 'episode_complete', 'no_oracle_target')


def dataset_fixture(tmp_path):
    """Structurally valid synthetic schema fixtures, not acquisition evidence."""
    episodes = []
    for number, split in enumerate(('train', 'validation', 'test')):
        episode_id = 'fixture_' + split
        n = 24
        state = np.full((n, 32), number * 5., np.float32)
        state[:, 0] += np.arange(n) * .01
        actions = np.zeros((n, 7), np.float32)
        actions[:, 0] = .00001
        phases = np.array(['APPROACH'] * 12 + ['RETRACT'] * 12)
        arrays = dict(state=state, actions=actions, rgb=np.full((n, 16, 24, 3), 100 + number, np.uint8),
                      times=np.arange(n) * .05, phases=phases, targets=np.array(['targetA'] * n),
                      prompts=np.array(['Approach targetA.'] * 12 + ['Retract from targetA.'] * 12),
                      target_sources=np.array(['rgbd_surface_tracking'] * n))
        path = tmp_path / (episode_id + '.npz')
        np.savez(path, **arrays)
        qualified = dict(schema='wam_vla_episode_qualification_v1', episode_id=episode_id,
                         arrays_sha256=digest(path), data_kind='closed_loop_rollout', qualified=True,
                         action_contract_version=api().ACTION_CONTRACT_VERSION,
                         action_contract_sha256=digest(Path(api().__file__).with_name('action_contract.py')),
                         modalities=MODALITIES, checks={name: True for name in CHECKS})
        qualification_path = tmp_path / (episode_id + '.qualification.json')
        write_json(qualification_path, qualified)
        episodes.append(dict(episode_id=episode_id, split=split, body_source='body_' + split,
                             layout_family='layout_' + split, motion_family='motion_' + split,
                             path=path.name, sha256=digest(path), qualification_path=qualification_path.name,
                             qualification_sha256=digest(qualification_path)))
    manifest = dict(schema='wam_vla_training_manifest_v1', sealed=True, modalities=MODALITIES,
                    heldout_axes=['body_source', 'layout_family', 'motion_family'],
                    action_period_s=.05, horizon=10, episodes=episodes)
    manifest_path = tmp_path / 'manifest.json'
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def rewrite_episode(root, manifest, number, mutation):
    record = manifest['episodes'][number]
    path = root / record['path']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mutation(arrays)
    np.savez(path, **arrays)
    record['sha256'] = digest(path)
    qualification_path = root / record['qualification_path']
    qualification = json.loads(qualification_path.read_text())
    qualification['arrays_sha256'] = record['sha256']
    write_json(qualification_path, qualification)
    record['qualification_sha256'] = digest(qualification_path)
    write_json(root / 'manifest.json', manifest)


def test_load_binds_hashes_train_only_norm_and_full_phase_chunks(tmp_path):
    module = api()
    manifest_path, manifest = dataset_fixture(tmp_path)
    dataset = module.load_dataset(manifest_path)
    assert dataset.normalization['fit_episode_ids'] == ['fixture_train']
    assert max(dataset.normalization['state']['high']) < 1.
    assert dataset.groups['heldout_axes'] == manifest['heldout_axes']
    assert len(dataset.indices['train']) == 6  # Three chunks per 12-step phase.
    assert [start for _, start in dataset.indices['train']] == [0, 1, 2, 12, 13, 14]
    assert len(dataset.data_hash) == 64


@pytest.mark.parametrize('mutation', ['unsealed', 'missing_validation', 'leak', 'audio', 'horizon', 'empty_axes'])
def test_manifest_rejects_unqualified_contracts(tmp_path, mutation):
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    if mutation == 'unsealed': manifest['sealed'] = False
    elif mutation == 'missing_validation': manifest['episodes'] = manifest['episodes'][:1]
    elif mutation == 'leak': manifest['episodes'][1]['body_source'] = 'body_train'
    elif mutation == 'audio': manifest['modalities'] = dict(MODALITIES, audio='microphone')
    elif mutation == 'horizon': manifest['horizon'] = 50
    else: manifest['heldout_axes'] = []
    write_json(path, manifest)
    with pytest.raises(ValueError): module.load_dataset(path)


@pytest.mark.parametrize('mutation', ['snapshot', 'unqualified', 'oracle', 'false_check', 'array_binding'])
def test_qualification_rejects_snapshot_or_oracle_declarations(tmp_path, mutation):
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    record = manifest['episodes'][0]
    qualification_path = tmp_path / record['qualification_path']
    qualification = json.loads(qualification_path.read_text())
    if mutation == 'snapshot': qualification['data_kind'] = 'snapshot'
    elif mutation == 'unqualified': qualification['qualified'] = False
    elif mutation == 'oracle': qualification['modalities'] = dict(MODALITIES, target='fakeoracle_target')
    elif mutation == 'false_check': qualification['checks']['no_oracle_target'] = False
    else: qualification['arrays_sha256'] = '0' * 64
    write_json(qualification_path, qualification)
    record['qualification_sha256'] = digest(qualification_path)
    write_json(path, manifest)
    with pytest.raises(ValueError): module.load_dataset(path)


@pytest.mark.parametrize('tamper',[None,'hash','version'])
def test_prior_stricter_same_semantics_qualification_is_explicitly_bound(tmp_path,tamper):
    module=api();path,manifest=dataset_fixture(tmp_path)
    record=manifest['episodes'][0];qualified=tmp_path/record['qualification_path']
    report=json.loads(qualified.read_text())
    version,sha=next(iter(module.COMPATIBLE_QUALIFICATIONS))
    report['action_contract_version']=version if tamper!='version' else 'unknown_contract'
    report['action_contract_sha256']=sha if tamper!='hash' else '0'*64
    write_json(qualified,report);record['qualification_sha256']=digest(qualified);write_json(path,manifest)
    if tamper:
        with pytest.raises(ValueError):module.load_dataset(path)
    else:assert module.load_dataset(path).indices['train']


@pytest.mark.parametrize('mutation', ['state7', 'rgbfloat', 'object_prompts', 'oracle_sources',
                                     'missing_modality', 'short_phase', 'nan', 'unsafe_action', 'prompt_change'])
def test_array_contract_rejects_malformed_and_nonchunks(tmp_path, mutation):
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    def change(arrays):
        if mutation == 'state7': arrays['state'] = arrays['state'][:, :7]
        elif mutation == 'rgbfloat': arrays['rgb'] = arrays['rgb'].astype(float)
        elif mutation == 'object_prompts': arrays['prompts'] = arrays['prompts'].astype(object)
        elif mutation == 'oracle_sources': arrays['target_sources'] = np.array(['fakeoracle_target'] * 24)
        elif mutation == 'missing_modality': del arrays['target_sources']
        elif mutation == 'short_phase': arrays['phases'] = np.array(['APPROACH', 'RETRACT'] * 12)
        elif mutation == 'nan': arrays['state'][0, 0] = np.nan
        elif mutation == 'unsafe_action': arrays['actions'][:, 0] = .9
        else: arrays['prompts'][1] = 'Other instruction.'
    rewrite_episode(tmp_path, manifest, 0, change)
    with pytest.raises(ValueError): module.load_dataset(path)


@pytest.mark.parametrize('origin_s', [0., 100.])
def test_dataset_rejects_float32_seconds_with_explicit_precision_contract(tmp_path, origin_s):
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    def change(arrays):
        arrays['times'] = (origin_s + arrays['times']).astype(np.float32)
    rewrite_episode(tmp_path, manifest, 0, change)
    with pytest.raises(ValueError, match='float64'):
        module.load_dataset(path)


def test_payload_tampering_and_path_escape_rejected(tmp_path):
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    with (tmp_path / manifest['episodes'][0]['path']).open('ab') as target: target.write(b'changed')
    with pytest.raises(ValueError, match='hash'): module.load_dataset(path)
    manifest['episodes'][0]['path'] = '../outside.npz'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='path'): module.load_dataset(path)


class Tokens:
    def __init__(self): self.seen = []
    def encode(self, text, add_bos=True):
        self.seen.append(text)
        return list(range(40))


def test_model_batch_preserves_32_state_tokens_7_actions_one_rgb(tmp_path):
    module = api()
    path, _ = dataset_fixture(tmp_path)
    dataset = module.load_dataset(path)
    tokenizer = Tokens()
    observation, actions = module.make_batch(dataset, 'train', [0, 4], tokenizer)
    assert observation['state'].shape == (2, 32)
    assert actions.shape == (2, 10, 7)
    assert observation['images']['base_0_rgb'].shape == (2, 224, 224, 3)
    assert observation['image_masks']['base_0_rgb'].all()
    assert not observation['image_masks']['left_wrist_0_rgb'].any()
    assert len(tokenizer.seen[0].split('State: ')[1].split(';')[0].split()) == 32
    assert 'Phase RETRACT.' in tokenizer.seen[1]


def test_pretrained_merge_only_slices_task_projections_and_initializes_lora():
    module = api()
    pretrained = {('action_in_proj', 'kernel'): np.ones((32, 5)),
                  ('action_out_proj', 'kernel'): np.ones((5, 32)),
                  ('action_out_proj', 'bias'): np.ones(32),
                  ('PaliGemma', 'llm', 'kernel'): np.ones((4, 5))}
    reference = dict(pretrained)
    reference.update({('action_in_proj', 'kernel'): np.empty((7, 5)),
                      ('action_out_proj', 'kernel'): np.empty((5, 7)),
                      ('action_out_proj', 'bias'): np.empty(7),
                      ('PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'lora_a'): np.empty((4, 2))})
    merged = module.merge_pretrained(reference, pretrained, lambda path, shape: np.zeros(shape, np.float32))
    assert merged[('action_in_proj', 'kernel')].shape == (7, 5)
    assert merged[('action_in_proj', 'kernel')].dtype == np.float32
    assert merged[('PaliGemma', 'llm', 'kernel')] is pretrained[('PaliGemma', 'llm', 'kernel')]
    reference[('new_unknown',)] = np.empty(2)
    with pytest.raises(ValueError, match='missing'): module.merge_pretrained(reference, pretrained, lambda p, s: np.zeros(s))


def test_merge_initializes_all_twenty_real_official_lora_parameter_paths():
    from sim.auscultation_vla_v2.tests.test_openpi_adapter import OFFICIAL_LORA_PATHS
    module = api()
    pretrained = {('action_in_proj', 'kernel'): np.ones((32, 5)),
                  ('action_out_proj', 'kernel'): np.ones((5, 32)),
                  ('action_out_proj', 'bias'): np.ones(32)}
    reference = {('action_in_proj', 'kernel'): np.empty((7, 5)),
                 ('action_out_proj', 'kernel'): np.empty((5, 7)),
                 ('action_out_proj', 'bias'): np.empty(7)}
    expected = {tuple(path.split('/')) for path in OFFICIAL_LORA_PATHS}
    reference.update({path: np.empty((4, 2)) for path in expected})
    initialized = []
    def initialize(path, shape):
        initialized.append(path)
        return np.zeros(shape, np.float32)
    merged = module.merge_pretrained(reference, pretrained, initialize)
    assert set(initialized) == expected and len(initialized) == 20
    assert all(merged[path].dtype == np.float32 for path in expected)
    reference[('PaliGemma', 'llm', 'lora_diagnostics', 'kernel')] = np.empty((4, 2))
    with pytest.raises(ValueError, match='missing'):
        module.merge_pretrained(reference, pretrained, initialize)


def checkpoint_metadata():
    return dict(data_hash='a' * 64, config={'seed': 42, 'batch_size': 1},
                normalization={'fit_episode_ids': ['train']},
                sampler_rng_state=np.random.default_rng(42).bit_generator.state,
                best_validation_loss=1.5, best_step=4, heldout_axes=['body_source'])


def test_atomic_resume_retains_optimizer_rng_norm_config_and_data_hash(tmp_path):
    module = api()
    metadata = checkpoint_metadata()
    arrays = {'trainable_0': np.ones((3, 2), np.float32), 'optimizer_0': np.array(8, np.int32),
              'optimizer_1': np.full((3, 2), .25, np.float32), 'jax_rng': np.array([1, 2], np.uint32)}
    path = module.save_checkpoint(tmp_path, step=8, arrays=arrays, metadata=metadata)
    loaded, info = module.load_checkpoint(path, data_hash=metadata['data_hash'],
                                         config=metadata['config'], normalization=metadata['normalization'])
    assert info['step'] == 8 and info['sampler_rng_state'] == metadata['sampler_rng_state']
    for key in arrays: np.testing.assert_array_equal(loaded[key], arrays[key])
    assert json.loads((tmp_path / 'latest.json').read_text())['directory'] == path.name
    with pytest.raises(ValueError, match='data'): module.load_checkpoint(path, data_hash='b' * 64,
        config=metadata['config'], normalization=metadata['normalization'])
    with pytest.raises(ValueError, match='config'): module.load_checkpoint(path, data_hash=metadata['data_hash'],
        config={'seed': 0}, normalization=metadata['normalization'])


def test_resume_rejects_tampered_and_partial_checkpoint(tmp_path):
    module = api()
    metadata = checkpoint_metadata()
    path = module.save_checkpoint(tmp_path, step=1, arrays={'jax_rng': np.array([1, 2], np.uint32)}, metadata=metadata)
    with (path / 'arrays.npz').open('ab') as output: output.write(b'changed')
    with pytest.raises(ValueError, match='hash'): module.load_checkpoint(path, data_hash=metadata['data_hash'],
        config=metadata['config'], normalization=metadata['normalization'])
    partial = tmp_path / '.partial'
    partial.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)): module.load_checkpoint(partial,
        data_hash=metadata['data_hash'], config=metadata['config'], normalization=metadata['normalization'])


def test_runtime_pin_and_import_location_validation_without_gpu(tmp_path, monkeypatch):
    module = api()
    repo = tmp_path / 'openpi'
    source = repo / 'src/openpi/models/pi0_config.py'
    source.parent.mkdir(parents=True)
    source.touch()
    def command(args, **kwargs):
        return module.OPENPI_COMMIT + '\n' if 'rev-parse' in args else ''
    monkeypatch.setattr(module.subprocess, 'check_output', command)
    module.verify_openpi_checkout(repo, source)
    with pytest.raises(ValueError, match='outside'): module.verify_openpi_checkout(repo, tmp_path / 'other.py')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'wrong\n')
    with pytest.raises(ValueError, match='commit'): module.verify_openpi_checkout(repo, source)


def test_resume_finds_complete_promoted_checkpoint_when_pointer_was_not_written(tmp_path):
    module = api()
    metadata = checkpoint_metadata()
    arrays = {'jax_rng': np.array([1, 2], np.uint32)}
    first = module.save_checkpoint(tmp_path, step=5, arrays=arrays, metadata=metadata)
    second = module.save_checkpoint(tmp_path, step=6, arrays=arrays, metadata=metadata)
    # Model power loss between directory promotion and pointer update.
    write_json(tmp_path / 'latest.json', {'directory': first.name})
    incomplete = tmp_path / 'step-000000000007'
    incomplete.mkdir()
    assert module.resolve_resume(tmp_path, 'latest') == second


def test_validate_only_uses_cpu_and_never_claims_acquisition_verified(tmp_path, capsys):
    module = api()
    path, _ = dataset_fixture(tmp_path)
    result = module.main(['--manifest', str(path), '--validate-only'])
    assert result['task_trained'] is False
    assert result['acquisition_quality_verified'] is False
    assert 'external attestation' in result['groups']['qualification_scope']
    assert not (tmp_path / 'run.json').exists()


def test_batch_rejects_fractional_index_instead_of_silently_truncating(tmp_path):
    module = api()
    path, _ = dataset_fixture(tmp_path)
    dataset = module.load_dataset(path)
    with pytest.raises(ValueError): module.make_batch(dataset, 'train', [.5], Tokens())


@pytest.mark.parametrize('value', [[], {'schema': 'wam_vla_training_manifest_v1', 'sealed': True,
    'modalities': MODALITIES, 'horizon': 10, 'action_period_s': .05, 'heldout_axes': ['body_source'],
    'episodes': ['invalid-record']}])
def test_malformed_manifest_containers_fail_cleanly(tmp_path, value):
    module = api()
    path = tmp_path / 'bad.json'
    write_json(path, value)
    with pytest.raises(ValueError): module.load_dataset(path)


def test_recovery_repairs_stale_best_pointer_from_verified_checkpoint_metadata(tmp_path):
    module = api()
    arrays = {'jax_rng': np.array([1, 2], np.uint32)}
    first_metadata = dict(checkpoint_metadata(), best_step=5, best_validation_loss=2.)
    first = module.save_checkpoint(tmp_path, step=5, arrays=arrays, metadata=first_metadata)
    second_metadata = dict(first_metadata, best_step=6, best_validation_loss=1.)
    second = module.save_checkpoint(tmp_path, step=6, arrays=arrays, metadata=second_metadata)
    write_json(tmp_path / 'best.json', {'directory': first.name, 'step': 5, 'validation_loss': 2.})
    _, resumed = module.load_checkpoint(second, data_hash=second_metadata['data_hash'],
        config=second_metadata['config'], normalization=second_metadata['normalization'])
    module.restore_best_pointer(tmp_path, resumed)
    pointer = json.loads((tmp_path / 'best.json').read_text())
    assert pointer['directory'] == second.name and pointer['step'] == 6
    assert pointer['validation_loss'] == 1.
    assert pointer['metadata_sha256'] == digest(second / 'metadata.json')


@pytest.mark.parametrize('failure', ['missing', 'tampered'])
def test_recovery_never_publishes_missing_or_tampered_best_checkpoint(tmp_path, failure):
    module = api()
    arrays = {'jax_rng': np.array([1, 2], np.uint32)}
    metadata = dict(checkpoint_metadata(), best_step=5, best_validation_loss=1., step=6)
    if failure == 'tampered':
        path = module.save_checkpoint(tmp_path, step=5, arrays=arrays,
            metadata=dict(metadata, step=5))
        with (path / 'arrays.npz').open('ab') as output: output.write(b'changed')
    with pytest.raises((ValueError, FileNotFoundError)):
        module.restore_best_pointer(tmp_path, metadata)
    assert not (tmp_path / 'best.json').exists()


def test_canonical_phase_and_target_reach_both_batch_and_token_preflight(tmp_path):
    from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
    module = api()
    path, manifest = dataset_fixture(tmp_path)
    def generic_prompt(arrays):
        arrays['prompts'] = np.array(['Auscultate targetA.'] * len(arrays['times']))
    rewrite_episode(tmp_path, manifest, 0, generic_prompt)
    dataset = module.load_dataset(path)
    tokenizer = Tokens()
    module.make_batch(dataset, 'train', [0, 3], tokenizer)
    approach_text, retract_text = tokenizer.seen
    assert INSTRUCTIONS['APPROACH'] in approach_text and 'APPROACH' in approach_text
    assert INSTRUCTIONS['RETRACT'] in retract_text and 'RETRACT' in retract_text
    assert 'Target identifier: "targetA"' in approach_text
    assert 'Auscultate targetA.' not in approach_text
    preflight = Tokens()
    module.validate_token_budget(dataset, preflight)
    assert preflight.seen[0] == approach_text
    assert preflight.seen[3] == retract_text
    assert len(preflight.seen) == len(dataset.indices['train']) + len(dataset.indices['validation'])
