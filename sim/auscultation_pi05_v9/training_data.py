"""Conditional C action view over the unchanged qualified V8 B dataset.

Qualification reconstructs B's normalization through its original loader. C
preparation fits only when no saved normalization is supplied. Supplied C stats
are validated against B and used exactly, without a C refit. Runtime codec calls
need only these saved stats and an actual measured observation anchor.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v8 import training_data as parent
from . import action_encoding as codec

C = codec.C
MANIFEST_SHA256 = parent.MANIFEST_SHA256
PARENT_CODEC_SHA256 = codec.PARENT_CODEC_SHA256
PARENT_DATA_SOURCE_SHA256 = '86dc1e157012fec6983a68ee90940d5718b716ee6cd3a83f8faef447cc761172'
SCHEMA = 'wam_rational_q4_dataset_normalization_v1'
_need = parent._need
_sha = parent._sha
_digest = parent._digest


def core_sources():
    inherited = parent.core_sources()
    _need(inherited.get('v8/action_encoding.py') == PARENT_CODEC_SHA256
          and inherited.get('v8/training_data.py') == PARENT_DATA_SOURCE_SHA256, 'Frozen B data/codec source changed')
    here = Path(__file__).parent
    return dict(inherited, **{'v9/'+name:_sha(here/name) for name in ('action_encoding.py','training_data.py')})


@dataclass(frozen=True)
class Dataset:
    parent: object
    mode: str
    normalization: dict
    data_hash: str
    source_sha256: dict

    @property
    def episodes(self): return self.parent.episodes
    @property
    def indices(self): return self.parent.indices
    @property
    def dynamics(self): return self.parent.dynamics
    @property
    def manifest(self): return self.parent.manifest
    def rgb(self, episode_index): return self.parent.rgb(episode_index)


def _validate_normalization(base, norm, mode, expected_sha256=None):
    _need(isinstance(norm, dict) and norm.get('schema') == SCHEMA and norm.get('mode') == mode
          and norm.get('recipe') == codec.recipe(mode), 'Wrong C dataset normalization identity')
    _need(norm.get('state') == base.normalization['state']
          and norm.get('encoding_binding') == base.normalization['encoding_binding'], 'Original observation normalization changed')
    _need(norm.get('action_encoding_binding') == codec.action_encoding_binding(mode), 'C action source changed')
    _need(norm.get('parent_B_dataset_normalization_sha256') == _digest(base.normalization), 'B dataset normalization parent differs')
    stats = norm.get('residual_actions', {})
    codec.validate_stats(stats, mode=mode)
    _need(stats['parent_B_stats'] == base.normalization['residual_actions'], 'Saved C normalization refers to another B fit')
    records = parent._records(base)
    windows = [dict(episode_id=e['episode_id'], starts=list(map(int,e['starts']))) for e in records]
    _need(stats['fit_windows_sha256'] == _digest(windows)
          and stats['fit_episode_ids'] == [e['episode_id'] for e in records]
          and stats['fit_window_count'] == sum(len(e['starts']) for e in records), 'Original TRAIN window identity changed')
    actual = _digest(norm)
    _need(expected_sha256 is None or actual == expected_sha256, 'Saved C dataset normalization hash differs')
    return actual


def view_parent(base, mode=C, *, normalization=None, expected_normalization_sha256=None):
    """Pure view of already qualified B; never certifies a new acquisition."""
    codec.action_encoding_binding(mode)
    _need(isinstance(base, parent.Dataset) and base.mode == codec.parent.B, 'Qualified B dataset view required')
    parent.dataset_binding(base)
    sources = core_sources()
    if normalization is None:
        _need(expected_normalization_sha256 is None, 'Expected saved normalization requires supplied bytes')
        stats = codec.fit_stats(parent._records(base), mode, parent_stats=base.normalization['residual_actions'])
        normalization = dict(schema=SCHEMA, recipe=codec.recipe(mode), mode=mode,
            state=base.normalization['state'], encoding_binding=base.normalization['encoding_binding'],
            residual_actions=stats, action_encoding_binding=codec.action_encoding_binding(mode),
            parent_B_dataset_normalization_sha256=_digest(base.normalization))
    _validate_normalization(base, normalization, mode, expected_normalization_sha256)
    data_hash = _digest(dict(parent_data_hash=base.data_hash, recipe=codec.recipe(mode), mode=mode,
                             normalization=normalization, source_sha256=sources))
    return Dataset(base, mode, normalization, data_hash, sources)


def load_dataset(manifest, mode=C, *, normalization=None, expected_normalization_sha256=None):
    """Original full qualification; saved C normalization skips C fitting."""
    codec.action_encoding_binding(mode)
    core_sources()
    base = parent.load_dataset(manifest, codec.parent.B)
    return view_parent(base, mode, normalization=normalization,
                       expected_normalization_sha256=expected_normalization_sha256)


load_data = load_dataset


def dataset_binding(dataset):
    _need(isinstance(dataset, Dataset), 'Explicit C dataset required')
    parent.dataset_binding(dataset.parent)
    _need(dataset.source_sha256 == core_sources(), 'C dataset source changed')
    norm_hash = _validate_normalization(dataset.parent, dataset.normalization, dataset.mode)
    expected = _digest(dict(parent_data_hash=dataset.parent.data_hash, recipe=codec.recipe(dataset.mode),
                            mode=dataset.mode, normalization=dataset.normalization, source_sha256=dataset.source_sha256))
    _need(dataset.data_hash == expected, 'C dataset normalization/data hash differs')
    stats = dataset.normalization['residual_actions']
    return dict(schema='wam_rational_q4_dataset_binding_v1', recipe=codec.recipe(dataset.mode), mode=dataset.mode,
        parent_data_hash=dataset.parent.data_hash, data_hash=dataset.data_hash, normalization_sha256=norm_hash,
        residual_stats_sha256=codec.normalization_digest(stats, mode=dataset.mode),
        fit_windows_sha256=stats['fit_windows_sha256'], fit_window_count=stats['fit_window_count'],
        source_sha256=dataset.source_sha256, parent_action_encoding_binding=codec.parent.action_encoding_binding(codec.parent.B))


def validate_binding(dataset, expected):
    actual = dataset_binding(dataset)
    _need(isinstance(expected, dict) and expected == actual, 'Wrong C dataset binding')
    return actual


def make_batch(dataset, split, indices, tokenizer):
    _need(isinstance(dataset, Dataset), 'Explicit C dataset required')
    codec._mode(dataset.mode)
    # Delegate original index validation and every observation byte to B.
    observations, actions = parent.make_batch(dataset.parent, split, indices, tokenizer)
    for i, index in enumerate(indices):
        ei, start = dataset.indices[split][int(index)]
        episode = dataset.episodes[ei]
        encoded = codec.encode_chunk(episode['actions'][start:start+10], episode['state'][start,:7],
            str(episode['phases'][start]), dataset.normalization['residual_actions'], mode=dataset.mode)
        _need(actions[i][:,codec.SIX].tobytes() == encoded[:,codec.SIX].tobytes(), 'Six B encoded coordinates changed')
        actions[i,:,3] = encoded[:,3]
    return observations, actions


def decode_prediction(dataset, normalized, state, phase):
    _need(isinstance(dataset, Dataset), 'Explicit C dataset required')
    raw = codec._array(state, (41,), 'original observation state')
    return codec.decode_chunk(normalized, raw[:7], phase, dataset.normalization['residual_actions'], mode=dataset.mode)
