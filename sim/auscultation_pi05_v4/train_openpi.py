"""Pinned official pi0.5 LoRA + seven-channel task-projection training entrypoint.

Run with ``python -m sim.auscultation_vla_v2.train_openpi --manifest ... --output
... --validate-only`` for CPU contract validation. Actual training requires the
existing /home/william/wam-vla-v2/vendor/openpi/.venv environment and a GPU; this
module never creates a dataset, downloads weights, or connects to another host.

Dataset contract (all paths relative to the manifest directory):
* manifest schema wam_vla_training_manifest_v1; sealed=true; exact MODALITIES;
  heldout_axes; horizon=10; action_period_s=.05; episodes list containing
  episode_id, split, body_source, layout_family, motion_family, path, sha256,
  qualification_path, qualification_sha256. Train AND validation are required.
* NPZ arrays: float state[N,32], actions[N,7]; float64 times[N] in seconds;
  uint8 rgb[N,H,W,3]. Timestamps must retain float64 precision from acquisition;
  upcasting rounded float32 values cannot recover it. Cadence tolerance stays
  fixed at 1e-7 seconds so absolute clock magnitude never relaxes gap rejection.
  Unicode phases[N], targets[N], prompts[N], target_sources[N]. No objects or
  pickle. Prompts are constant within each phase/target combination. Sources
  must be rgbd_surface_tracking. Only complete same-phase/target 10-step chunks
  at 20 Hz are used. No padding, interpolation, or snapshot training fallback.
  Raw prompts remain provenance metadata; actual training instructions come
  from the canonical phase instruction plus the opaque target identifier.
* qualification schema wam_vla_episode_qualification_v1; matching episode_id
  and arrays_sha256; data_kind=closed_loop_rollout; qualified=true; exact
  MODALITIES; every QUALIFICATION_CHECKS entry exactly true.

Seals/qualification flags are EXTERNAL ATTESTATIONS, not proof of acquisition
quality. Independent collector audit must establish their evidence. Structural
validation cannot identify deliberately mislabeled synthetic/oracle arrays.
Checkpoints contain trainable parameters, complete optimizer state, JAX and
sampler RNG state, step, normalization, config, data hash, and split scope.
Frozen parameters are restored from the reverified public checkpoint on resume.
SIGTERM/SIGINT request a checkpoint after the current complete optimizer step;
hard termination can lose only work after the last complete checkpoint. This is
best-effort preemption handling, not a guarantee against a short eviction grace.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

import numpy as np

from sim.auscultation_vla_v2.chunks import PHASES
from .action_contract import bounded,VERSION as ACTION_CONTRACT_VERSION,COMPATIBLE_QUALIFICATIONS
from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS
from .openpi_adapter import (OPENPI_COMMIT, adapt_projection_params, prepare_images,
                             lora_path, tokenize_state, trainable_path)
from .training_data import (MODALITIES, fit_normalization, full_chunks, normalize,
                            validate_groups, validate_modalities, action_stats, phase_balanced_indices)

ROOT = Path('/home/william/wam-vla-v2')
QUALIFICATION_CHECKS = ('rgbd_tracking_valid', 'measured_state_valid', 'hall_valid',
                        'synchronized', 'actions_bounded', 'episode_complete', 'no_oracle_target')
ARRAY_KEYS = {'state', 'actions', 'rgb', 'times', 'phases', 'targets', 'prompts', 'target_sources'}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _bound_file(root, relative, expected_hash):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '..' in Path(relative).parts):
        raise ValueError('invalid relative artifact path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('artifact path outside sealed root')
    if path.is_symlink() or not path.is_file():
        raise ValueError('artifact path must be a regular file')
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or _sha(path) != expected_hash):
        raise ValueError('artifact hash mismatch: ' + relative)
    return path


@dataclass
class Dataset:
    manifest: dict
    episodes: list
    indices: dict
    normalization: dict
    groups: dict
    data_hash: str
    _rgb_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def rgb(self, episode_index):
        # NPZ cannot mmap compressed images. Bound memory to two complete image
        # episodes; scalar state/action arrays remain in RAM. Rehash on reload.
        if episode_index not in self._rgb_cache:
            episode = self.episodes[episode_index]
            if _sha(episode['_path']) != episode['sha256']:
                raise ValueError('episode hash changed after validation')
            with np.load(episode['_path'], allow_pickle=False) as archive:
                self._rgb_cache[episode_index] = archive['rgb']
            if len(self._rgb_cache) > 2:
                self._rgb_cache.popitem(last=False)
        self._rgb_cache.move_to_end(episode_index)
        return self._rgb_cache[episode_index]


def _validate_arrays(arrays):
    if set(arrays) != ARRAY_KEYS:
        raise ValueError('episode array keys do not match modality contract')
    state, actions, rgb, times = (arrays[key] for key in ('state', 'actions', 'rgb', 'times'))
    n = len(state) if state.ndim else 0
    for name, shape in [('state', (n, 32)), ('actions', (n, 7)), ('times', (n,))]:
        value = arrays[name]
        if (value.shape != shape or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()):
            raise ValueError('invalid episode ' + name)
    if times.dtype.kind != 'f' or times.dtype.itemsize != 8:
        raise ValueError('episode times must use float64 seconds')
    if n < 10 or rgb.ndim != 4 or rgb.shape[0] != n or rgb.shape[3] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[1:3]) < 2:
        raise ValueError('invalid episode RGB sequence')
    for name in ('phases', 'targets', 'prompts', 'target_sources'):
        value = arrays[name]
        if value.shape != (n,) or value.dtype.kind != 'U' or any(not text.strip() for text in value.tolist()):
            raise ValueError('invalid Unicode episode ' + name)
    if np.any(arrays['target_sources'] != 'rgbd_surface_tracking'):
        raise ValueError('oracle or unsupported target source')
    if not set(arrays['phases']) <= PHASES:
        raise ValueError('invalid episode phase')
    prompts = {}
    for phase, target, prompt in zip(arrays['phases'], arrays['targets'], arrays['prompts']):
        identity = (phase, target)
        if identity in prompts and prompts[identity] != prompt:
            raise ValueError('prompt changed within a phase/target')
        prompts[identity] = prompt
    if any(not bounded(actions[arrays['phases']==phase],phase) for phase in set(arrays['phases'])):
        raise ValueError('episode action exceeds engineering proposal bounds')
    chunks = full_chunks(times, arrays['phases'].tolist(), arrays['targets'].tolist(), actions,
                         horizon=10, period_s=.05)
    if not chunks:
        raise ValueError('episode has no complete ten-step chunks')
    return [chunk['start'] for chunk in chunks]


def load_dataset(manifest_path):
    """Verify external attestations/bindings; do not certify acquisition quality."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if (not isinstance(manifest, dict)
            or manifest.get('schema') != 'wam_vla_training_manifest_v1' or manifest.get('sealed') is not True
            or type(manifest.get('horizon')) is not int or manifest['horizon'] != 10
            or manifest.get('action_period_s') != .05):
        raise ValueError('sealed ten-step 20-Hz training manifest required')
    validate_modalities(manifest.get('modalities'))
    if 'protocol_sha256' in manifest:
        _bound_file(manifest_path.parent,'protocol.json',manifest['protocol_sha256'])
    records = manifest.get('episodes')
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError('manifest episodes must be a nonempty record list')
    groups = validate_groups(records, heldout_axes=manifest.get('heldout_axes'))
    if not {'train', 'validation'} <= {record['split'] for record in records}:
        raise ValueError('separate train and validation episodes are required')
    episodes, indices, hashes = [], {key: [] for key in ('train', 'validation', 'test')}, set()
    for episode_index, record in enumerate(records):
        path = _bound_file(manifest_path.parent, record.get('path'), record.get('sha256'))
        report_path = _bound_file(manifest_path.parent, record.get('qualification_path'), record.get('qualification_sha256'))
        report = _read_json(report_path)
        if (report.get('schema') != 'wam_vla_episode_qualification_v1'
                or report.get('episode_id') != record['episode_id']
                or report.get('arrays_sha256') != record['sha256']
                or report.get('data_kind') != 'closed_loop_rollout' or report.get('qualified') is not True
                or (report.get('action_contract_version'),report.get('action_contract_sha256')) not in
                    (COMPATIBLE_QUALIFICATIONS | {(ACTION_CONTRACT_VERSION,_sha(Path(__file__).with_name('action_contract.py')))})
                or any(report.get('checks', {}).get(key) is not True for key in QUALIFICATION_CHECKS)):
            raise ValueError('qualified closed-loop episode attestation required')
        validate_modalities(report.get('modalities'))
        if record['sha256'] in hashes:
            raise ValueError('duplicate episode content cannot be a new group')
        hashes.add(record['sha256'])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        starts = _validate_arrays(arrays)
        del arrays['rgb']
        episodes.append(dict(record, **arrays, _path=path))
        indices[record['split']].extend((episode_index, start) for start in starts)
    normalization = fit_normalization([episode for episode in episodes if episode['split'] == 'train'])
    groups['qualification_scope'] = 'external attestation; acquisition evidence requires independent collector audit'
    data_hash = hashlib.sha256(_json_bytes(dict(manifest_sha256=_sha(manifest_path), manifest=manifest))).hexdigest()
    return Dataset(manifest, episodes, indices, normalization, groups, data_hash)


def training_instruction(phase, target):
    """Bind the supervised phase without inventing anatomical target semantics."""
    if phase not in INSTRUCTIONS or not isinstance(target, str) or not target.strip():
        raise ValueError('invalid instruction phase or target identity')
    return f'Phase {phase}. {INSTRUCTIONS[phase]} Target identifier: {json.dumps(target, ensure_ascii=False)}.'


def validate_token_budget(dataset, tokenizer):
    """Use the exact same canonical instructions as batches before model load."""
    for split in ('train', 'validation'):
        for episode_index, start in dataset.indices[split]:
            episode = dataset.episodes[episode_index]
            tokenize_state(tokenizer, training_instruction(str(episode['phases'][start]), str(episode['targets'][start])),
                           normalize(episode['state'][start], dataset.normalization['state']))


def make_batch(dataset, split, indices, tokenizer):
    """CPU preprocessing; retain every state channel independently of action_dim."""
    states, actions, tokens, token_masks, images, masks = [], [], [], [], {}, {}
    if split not in {'train', 'validation', 'test'} or len(indices) == 0:
        raise ValueError('invalid nonempty split batch')
    for index in indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(dataset.indices[split]):
            raise ValueError('batch index outside split')
        episode_index, start = dataset.indices[split][int(index)]
        episode = dataset.episodes[episode_index]
        state = normalize(episode['state'][start], dataset.normalization['state'])
        instruction = training_instruction(str(episode['phases'][start]), str(episode['targets'][start]))
        token, mask = tokenize_state(tokenizer, instruction, state, max_len=200)
        frame, present = prepare_images(dataset.rgb(episode_index)[start])
        states.append(state)
        actions.append(normalize(episode['actions'][start:start+10], action_stats(dataset.normalization,str(episode['phases'][start]))))
        tokens.append(token)
        token_masks.append(mask)
        for key in frame:
            images.setdefault(key, []).append(frame[key])
            masks.setdefault(key, []).append(present[key])
    return dict(images={key: np.stack(value) for key, value in images.items()},
                image_masks={key: np.asarray(value, bool) for key, value in masks.items()},
                state=np.stack(states), tokenized_prompt=np.stack(tokens),
                tokenized_prompt_mask=np.stack(token_masks)), np.stack(actions)


def merge_pretrained(reference, pretrained, initialize_lora):
    """Exact pretrained shape binding with only declared seven-channel slicing."""
    loaded = adapt_projection_params(pretrained)
    merged = {}
    for path, spec in reference.items():
        if path in loaded:
            if loaded[path].shape != spec.shape:
                raise ValueError('pretrained shape mismatch: ' + str(path))
            merged[path] = loaded[path].astype(np.float32) if trainable_path(path) else loaded[path]
        elif lora_path(path):
            merged[path] = initialize_lora(path, spec.shape)
        else:
            raise ValueError('unexplained missing pretrained key: ' + str(path))
    if not any(lora_path(path) for path in merged):
        raise ValueError('LoRA factors missing from model')
    return merged


def verify_openpi_checkout(repo, imported_source):
    repo, source = Path(repo).resolve(), Path(imported_source).resolve()
    if not source.is_relative_to(repo / 'src' / 'openpi'):
        raise ValueError('imported openpi is outside pinned checkout')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != OPENPI_COMMIT:
        raise ValueError('official openpi commit mismatch')
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('official openpi tracked files are modified')
    return commit


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as output:
        output.write(_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def save_checkpoint(output, *, step, arrays, metadata):
    """Atomic immutable numeric-only checkpoint; partial directories are ignored."""
    output = Path(output)
    if type(step) is not int or step < 0 or not arrays:
        raise ValueError('invalid checkpoint step or arrays')
    for key, array in arrays.items():
        if not isinstance(key, str) or np.asarray(array).dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError('checkpoint arrays must be finite numeric values')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ('step-%012d' % step)
    if destination.exists():
        raise ValueError('checkpoint step already exists')
    temporary = Path(tempfile.mkdtemp(prefix='.partial-', dir=output))
    try:
        with (temporary / 'arrays.npz').open('wb') as target:
            np.savez(target, **arrays)
            target.flush()
            os.fsync(target.fileno())
        report = dict(metadata, schema='wam_pi05_resume_v1', step=step,
                      arrays_sha256=_sha(temporary / 'arrays.npz'))
        _atomic_json(temporary / 'metadata.json', report)
        _atomic_json(temporary / 'COMPLETE.json', {'metadata_sha256': _sha(temporary / 'metadata.json')})
        os.replace(temporary, destination)
        _atomic_json(output / 'latest.json', {'directory': destination.name,
                                             'metadata_sha256': _sha(destination / 'metadata.json')})
        descriptor = os.open(output, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return destination


def _verified_checkpoint_metadata(path, *, data_hash, config, normalization):
    path = Path(path)
    complete = _read_json(path / 'COMPLETE.json')
    if _sha(path / 'metadata.json') != complete.get('metadata_sha256'):
        raise ValueError('checkpoint metadata hash mismatch')
    report = _read_json(path / 'metadata.json')
    if report.get('schema') != 'wam_pi05_resume_v1' or type(report.get('step')) is not int or report['step'] < 0:
        raise ValueError('invalid checkpoint metadata')
    for key, expected in [('data_hash', data_hash), ('config', config), ('normalization', normalization)]:
        if report.get(key) != expected:
            raise ValueError('resume ' + key + ' mismatch')
    if _sha(path / 'arrays.npz') != report.get('arrays_sha256'):
        raise ValueError('checkpoint arrays hash mismatch')
    return report


def load_checkpoint(path, *, data_hash, config, normalization):
    path = Path(path)
    report = _verified_checkpoint_metadata(path, data_hash=data_hash, config=config, normalization=normalization)
    with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(value.dtype.kind not in 'biuf' or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('invalid numeric checkpoint arrays')
    return arrays, report


def restore_best_pointer(output, resumed):
    """Repair a preempted selection publication using hash-verified saved state."""
    output = Path(output)
    best_step, best_loss = resumed.get('best_step'), resumed.get('best_validation_loss')
    if best_step is None and best_loss is None:
        (output / 'best.json').unlink(missing_ok=True)
        return None
    if (type(best_step) is not int or not 0 <= best_step <= resumed['step']
            or not isinstance(best_loss, (int, float)) or not np.isfinite(best_loss)):
        raise ValueError('invalid resumed best-checkpoint selection')
    best_path = output / ('step-%012d' % best_step)
    best = _verified_checkpoint_metadata(best_path, data_hash=resumed['data_hash'],
                                          config=resumed['config'], normalization=resumed['normalization'])
    if (best['step'] != best_step or best.get('best_step') != best_step
            or best.get('best_validation_loss') != best_loss):
        raise ValueError('best checkpoint metadata does not match resumed selection')
    pointer = dict(directory=best_path.name, step=best_step, validation_loss=best_loss,
                   data_hash=resumed['data_hash'], metadata_sha256=_sha(best_path / 'metadata.json'))
    _atomic_json(output / 'best.json', pointer)
    return best_path


def resolve_resume(output, requested):
    """Find the newest committed directory, even if preemption left a stale pointer.

    Content/config checks still run in ``load_checkpoint``. A corrupt complete
    checkpoint is an error; it is never silently replaced by an older state.
    """
    if requested != 'latest':
        return Path(requested)
    complete = [path for path in Path(output).glob('step-*')
                if len(path.name) == 17 and path.name[5:].isdigit()
                and path.is_dir() and not path.is_symlink() and (path / 'COMPLETE.json').is_file()]
    if not complete:
        raise ValueError('no complete checkpoint available to resume')
    return max(complete, key=lambda path: path.name)


def _verify_public_assets(root):
    receipt_path = root / 'public_checkpoint_verified.json'
    receipt = _read_json(receipt_path)
    if (receipt.get('schema') != 'verified_public_checkpoint_v1'
            or receipt.get('source') != 'gs://openpi-assets/checkpoints/pi05_base' or not receipt.get('files')):
        raise ValueError('verified public checkpoint receipt required')
    asset_root = root / 'public-assets' / 'pi05_base'
    parameter_files = 0
    for record in receipt['files']:
        prefix = 'checkpoints/pi05_base/'
        if not record['name'].startswith(prefix):
            raise ValueError('unexpected public checkpoint artifact')
        relative = record['name'][len(prefix):]
        path = _bound_file(asset_root, relative, record['sha256'])
        if path.stat().st_size != record['bytes']:
            raise ValueError('public checkpoint size mismatch')
        parameter_files += relative.startswith('params/')
    if not parameter_files:
        raise ValueError('no verified public parameter files')
    tokenizer_receipt = root / 'public_tokenizer_verified.json'
    tokenizer_meta = _read_json(tokenizer_receipt)
    tokenizer = _bound_file(root / 'public-assets', 'paligemma_tokenizer.model', tokenizer_meta.get('sha256'))
    return asset_root / 'params', tokenizer, dict(checkpoint_receipt_sha256=_sha(receipt_path),
                                                tokenizer_sha256=_sha(tokenizer))


def train(dataset, args):
    """Actual official JAX flow-matching optimization. No CPU/model fallback."""
    repo = ROOT / 'vendor' / 'openpi'
    if Path(sys.prefix).resolve() != (repo / '.venv').resolve():
        raise RuntimeError('run training in the existing official openpi .venv')
    # Imports remain lazy so CPU schema and checkpoint checks need no JAX install.
    import flax
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    import sentencepiece
    from openpi.models import model as model_api
    from openpi.models.pi0_config import Pi0Config

    verify_openpi_checkout(repo, inspect.getfile(Pi0Config))
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('exactly one GPU required; no silent CPU fallback')
    params_path, tokenizer_path, assets = _verify_public_assets(ROOT)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    config = dict(openpi_commit=OPENPI_COMMIT, action_dim=7, action_horizon=10, state_width=32,
                  max_token_len=200, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                  batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate,
                  steps=args.steps, save_every=args.save_every, evaluate_every=args.evaluate_every,
                  optimizer='clip_global_norm(1)+Adam', jax=jax.__version__, flax=flax.__version__,
                  optax=optax.__version__, assets=assets,
                  canonical_phase_instructions=dict(INSTRUCTIONS), instruction_policy='canonical-phase-and-target-v1',
                  recipe='phase_balanced_phase_action_scaling_v4',state_encoding_binding=dataset.normalization['encoding_binding'],
                  qualification='research_only_archived_single_layout_data_not_closed_loop_validated',
                  adapter_sha256=_sha(Path(__file__).with_name('openpi_adapter.py')),
                  action_contract_version=ACTION_CONTRACT_VERSION,
                  action_contract_sha256=_sha(Path(__file__).with_name('action_contract.py')),
                  trainer_sha256=_sha(__file__), training_data_sha256=_sha(Path(__file__).with_name('training_data.py')))
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('nonempty output requires explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    # Verify tokens before allocating the pretrained model, including validation.
    validate_token_budget(dataset, tokenizer)
    cfg = Pi0Config(pi05=True, action_dim=7, action_horizon=10, max_token_len=200,
                    paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    skeleton = nnx.eval_shape(cfg.create, jax.random.key(args.seed))
    reference = traverse_util.flatten_dict(nnx.state(skeleton).to_pure_dict())
    pretrained = traverse_util.flatten_dict(model_api.restore_params(params_path, dtype=jnp.bfloat16))
    if not all(bool(jax.device_get(jnp.isfinite(value).all())) for value in pretrained.values()):
        raise ValueError('nonfinite public pretrained parameter')
    path_indices = {path: index for index, path in enumerate(reference)}
    merged = merge_pretrained(reference, pretrained,
        lambda path, shape: jax.random.normal(jax.random.fold_in(jax.random.key(args.seed), path_indices[path]),
                                             shape, jnp.float32) * .01)
    model = cfg.load(traverse_util.unflatten_dict(merged))
    model.train()
    train_filter = nnx.All(nnx.Param, lambda path, value: trainable_path(path))
    graph, trainable, frozen = nnx.split(model, train_filter, ...)
    actual_trainable = traverse_util.flatten_dict(trainable.to_pure_dict())
    if not actual_trainable or not all(trainable_path(path) for path in actual_trainable):
        raise ValueError('invalid trainable parameter partition')
    if not {'action_in_proj', 'action_out_proj'} <= {path[0] for path in actual_trainable}:
        raise ValueError('task projections were not selected for training')
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    optimizer_state = optimizer.init(trainable)
    del skeleton, pretrained, merged, model

    def objective(parameters, frozen_parameters, observation, actions, rng, training):
        candidate = nnx.merge(graph, parameters, frozen_parameters)
        candidate.train() if training else candidate.eval()
        return candidate.compute_loss(rng, observation, actions, train=training).mean()

    gradient = jax.jit(jax.value_and_grad(
        lambda p, f, o, a, r: objective(p, f, o, a, r, True), argnums=0))
    evaluate = jax.jit(lambda p, f, o, a, r: objective(p, f, o, a, r, False))

    @jax.jit
    def optimize(parameters, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, parameters)
        return optax.apply_updates(parameters, updates), new_opt_state

    train_phases=[str(dataset.episodes[e]['phases'][i]) for e,i in dataset.indices['train']]
    train_episodes=[e for e,i in dataset.indices['train']]
    sampler = np.random.default_rng(args.seed)
    rng = jax.random.key(args.seed)
    step, best_step, best_loss = 0, None, None
    tree_template = (trainable, optimizer_state)
    path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(tree_template)
    leaf_contract = [dict(path=jax.tree_util.keystr(path), shape=list(value.shape), dtype=str(value.dtype))
                     for path, value in path_leaves]
    if args.resume:
        resume_path = resolve_resume(output, args.resume)
        saved, metadata = load_checkpoint(resume_path, data_hash=dataset.data_hash,
                                           config=config, normalization=dataset.normalization)
        if metadata.get('leaf_contract') != leaf_contract:
            raise ValueError('resume model/optimizer tree mismatch')
        expected_keys = {'leaf_%05d' % index for index in range(len(path_leaves))} | {'jax_rng'}
        if set(saved) != expected_keys:
            raise ValueError('resume numeric leaf set mismatch')
        leaves = []
        for index, (_, expected) in enumerate(path_leaves):
            value = saved['leaf_%05d' % index]
            if value.shape != expected.shape or str(value.dtype) != str(expected.dtype):
                raise ValueError('resume numeric leaf shape/dtype mismatch')
            leaves.append(jnp.asarray(value))
        trainable, optimizer_state = jax.tree_util.tree_unflatten(tree_definition, leaves)
        rng = jax.random.wrap_key_data(jnp.asarray(saved['jax_rng']))
        sampler.bit_generator.state = metadata['sampler_rng_state']
        step, best_step, best_loss = metadata['step'], metadata['best_step'], metadata['best_validation_loss']
        restore_best_pointer(output, metadata)
    _atomic_json(output / 'run.json', dict(config=config, data_hash=dataset.data_hash,
                                          groups=dataset.groups, normalization=dataset.normalization,
                                          manifest=dataset.manifest, trainable_paths=['/'.join(path) for path in actual_trainable]))
    stop = [False]
    def stop_after_step(signum, frame):
        stop[0] = True
    previous_handlers = {sig: signal.signal(sig, stop_after_step) for sig in (signal.SIGTERM, signal.SIGINT)}
    last_saved = step if args.resume else -1

    def batch(split, indices):
        observation, actions = make_batch(dataset, split, indices, tokenizer)
        return jax.device_put(model_api.Observation(**observation)), jax.device_put(actions)

    def checkpoint():
        leaves = jax.tree.leaves((trainable, optimizer_state))
        arrays = {'leaf_%05d' % index: np.asarray(value) for index, value in enumerate(jax.device_get(leaves))}
        arrays['jax_rng'] = np.asarray(jax.random.key_data(rng))
        metadata = dict(data_hash=dataset.data_hash, config=config, normalization=dataset.normalization,
                        heldout_axes=dataset.manifest['heldout_axes'], split_scope=dataset.groups,
                        sampler_rng_state=sampler.bit_generator.state, best_validation_loss=best_loss,
                        best_step=best_step, leaf_contract=leaf_contract)
        return save_checkpoint(output, step=step, arrays=arrays, metadata=metadata)

    try:
        while step < args.steps and not stop[0]:
            selection = phase_balanced_indices(train_phases,args.batch_size,sampler,train_episodes)
            observation, actions = batch('train', selection)
            rng, loss_rng = jax.random.split(rng)
            loss, grads = gradient(trainable, frozen, observation, actions, loss_rng)
            finite_gradients = bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(value).all() for value in jax.tree.leaves(grads)]))))
            loss_value = float(loss)
            if not np.isfinite(loss_value) or not finite_gradients:
                raise ValueError('nonfinite training loss/gradient; last complete checkpoint remains resumable')
            trainable, optimizer_state = optimize(trainable, optimizer_state, grads)
            jax.block_until_ready((trainable, optimizer_state))
            step += 1
            metrics = dict(step=step, train_loss=loss_value)
            evaluate_now = step % args.evaluate_every == 0 or step == args.steps
            improved = False
            if evaluate_now:
                weighted, count = 0., 0
                for start in range(0, len(dataset.indices['validation']), args.batch_size):
                    indices = list(range(start, min(start + args.batch_size, len(dataset.indices['validation']))))
                    val_observation, val_actions = batch('validation', indices)
                    value = float(evaluate(trainable, frozen, val_observation, val_actions,
                                           jax.random.fold_in(jax.random.key(0), start)))
                    if not np.isfinite(value):
                        raise ValueError('nonfinite validation loss')
                    weighted += value * len(indices)
                    count += len(indices)
                metrics['validation_loss'] = weighted / count
                improved = best_loss is None or metrics['validation_loss'] < best_loss
                if improved:
                    best_loss, best_step = metrics['validation_loss'], step
            if step % args.save_every == 0 or evaluate_now or stop[0]:
                saved_path = checkpoint()
                last_saved = step
                if improved:
                    _atomic_json(output / 'best.json', dict(directory=saved_path.name, step=step,
                                                            validation_loss=best_loss, data_hash=dataset.data_hash,
                                                            metadata_sha256=_sha(saved_path / 'metadata.json')))
            print(json.dumps(metrics, allow_nan=False), flush=True)
        if last_saved != step:
            checkpoint()
        return dict(step=step, interrupted=stop[0], best_step=best_step, best_validation_loss=best_loss,
                    task_trained=step > 0, closed_loop_validated=False)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume', help='checkpoint directory or latest')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--evaluate-every', type=int, default=50)
    args = parser.parse_args(argv)
    if (min(args.steps, args.batch_size, args.save_every, args.evaluate_every) < 1 or args.seed < 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('positive training limits and a nonnegative seed are required')
    if not args.validate_only and args.output is None:
        parser.error('--output is required for training')
    dataset = load_dataset(args.manifest)
    report = dict(schema='wam_pi05_dataset_contract_report_v1', data_hash=dataset.data_hash,
                  chunks={key: len(value) for key, value in dataset.indices.items()}, groups=dataset.groups,
                  task_trained=False, acquisition_quality_verified=False)
    if not args.validate_only:
        report.update(train(dataset, args))
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
