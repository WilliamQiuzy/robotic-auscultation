"""Physical delta targets must preserve the immutable absolute-q supervision."""
import importlib
import json

import numpy as np
import pytest

from sim.auscultation_pi05_v6 import action_contract


def codec():
    try:
        module = importlib.import_module('sim.auscultation_pi05_v7.action_encoding')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'The V7 fixed-observation-anchor action codec is required'
    return module


def episode(*, phase='CONTACT', split='train', identity='training-a', count=20, starts=(0, 10)):
    anchor = (np.asarray(action_contract.LOWER) + np.asarray(action_contract.UPPER)) / 2
    state = np.zeros((count, 41), np.float32)
    state[:, :7] = anchor + np.arange(count)[:, None] * .0003
    actions = (state[:, :7] + np.linspace(.0001, .003, count)[:, None]).astype(np.float32)
    return dict(episode_id=identity, split=split, state=state, actions=actions,
                phases=np.full(count, phase), starts=list(starts))


def test_chunk_uses_one_observation_anchor_for_all_ten_future_commands():
    mod = codec()
    source = episode()
    stats = mod.fit_stats([source])
    anchor = source['state'][0, :7]
    commands = source['actions'][:10].copy()
    before = commands.copy()
    normalized = mod.encode_chunk(commands, anchor, 'CONTACT', stats)
    phase = stats['actions_by_phase']['CONTACT']
    reconstructed_delta = (normalized.astype(float) + 1) / 2 * (
        np.array(phase['high']) - phase['low']) + phase['low']
    np.testing.assert_allclose(reconstructed_delta, commands.astype(float) - anchor, atol=2e-8, rtol=0)
    assert not np.allclose(reconstructed_delta[-1], commands[-1] - source['state'][9, :7], atol=1e-5)
    decoded = mod.decode_chunk(normalized, anchor, 'CONTACT', stats)
    assert normalized.dtype == np.float32 and decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, before, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(commands, before)


def test_only_explicit_valid_starts_fit_statistics_and_record_physical_support():
    mod = codec()
    source = episode(count=30, starts=(0, 20))
    # Rows 10..19 represent omitted windows, e.g. excluded by the source cadence gate.
    source['actions'][10:20] += 1
    source['phases'][20:] = 'HOLD'
    saved = source['actions'].copy()
    stats = mod.fit_stats([source])
    assert stats['fit_episode_ids'] == ['training-a']
    assert stats['units'] == 'rad' and stats['horizon'] == 10
    assert stats['minimum_full_span_rad'] == .002
    assert stats['action_encoding_binding'] == mod.action_encoding_binding()
    for phase, start in [('CONTACT', 0), ('HOLD', 20)]:
        own = stats['actions_by_phase'][phase]
        expected = source['actions'][start:start+10].astype(float) - source['state'][start, :7]
        np.testing.assert_allclose(own['observed_min'], expected.min(0), atol=0, rtol=0)
        np.testing.assert_allclose(own['observed_max'], expected.max(0), atol=0, rtol=0)
        assert own['window_count'] == 1
    np.testing.assert_array_equal(source['actions'], saved)
    json.dumps(stats, allow_nan=False)


def test_constant_phase_retains_engineering_span():
    mod = codec()
    source = episode(count=10, starts=(0,))
    source['actions'][:] = source['state'][0, :7]
    stats = mod.fit_stats([source])
    own = stats['actions_by_phase']['CONTACT']
    np.testing.assert_allclose(np.array(own['high']) - own['low'], .002, atol=1e-15)
    np.testing.assert_allclose(mod.encode_chunk(source['actions'], source['state'][0, :7], 'CONTACT', stats), 0)


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_rejects_heldout_normalization_fit(split):
    with pytest.raises(ValueError, match='train'):
        codec().fit_stats([episode(split=split)])


@pytest.mark.parametrize('starts', [[], [True], [-1], [11], [0, 0], [0.0], np.array(0), [[0]]])
def test_rejects_invalid_or_duplicate_window_starts(starts):
    with pytest.raises(ValueError, match='start|window'):
        source = episode()
        source['starts'] = starts
        codec().fit_stats([source])


def test_rejects_phase_crossing_window_and_repeated_episode():
    mod = codec()
    source = episode()
    source['phases'][5:] = 'HOLD'
    with pytest.raises(ValueError, match='phase'):
        mod.fit_stats([source])
    with pytest.raises(ValueError, match='episode'):
        mod.fit_stats([episode(), episode()])


@pytest.mark.parametrize('value', [np.zeros((9, 7)), np.zeros((10, 8)), np.full((10, 7), np.nan),
                                  np.full((10, 7), np.inf), np.full((10, 7), '1')])
def test_rejects_invalid_chunk_shapes_or_values(value):
    mod = codec()
    source = episode()
    stats = mod.fit_stats([source])
    for transform in (mod.encode_chunk, mod.decode_chunk):
        with pytest.raises(ValueError):
            transform(value, source['state'][0, :7], 'CONTACT', stats)


@pytest.mark.parametrize('anchor', [np.zeros((1, 7)), np.zeros(6), np.full(7, np.nan)])
def test_rejects_invalid_observation_anchor(anchor):
    mod = codec()
    source = episode()
    stats = mod.fit_stats([source])
    for transform in (mod.encode_chunk, mod.decode_chunk):
        with pytest.raises(ValueError):
            transform(source['actions'][:10], anchor, 'CONTACT', stats)


def test_out_of_training_support_is_not_clipped_and_raw_urdf_gate_remains_separate():
    mod = codec()
    source = episode()
    stats = mod.fit_stats([source])
    anchor = source['state'][0, :7]
    commands = np.tile(np.asarray(action_contract.UPPER, np.float32) + .1, (10, 1))
    normalized = mod.encode_chunk(commands, anchor, 'CONTACT', stats)
    assert np.max(normalized) > 1
    decoded = mod.decode_chunk(normalized, anchor, 'CONTACT', stats)
    np.testing.assert_allclose(decoded, commands, atol=2e-6, rtol=0)
    assert not action_contract.bounded(decoded, 'CONTACT')


def test_random_full_robot_domain_roundtrip_matches_original_float32_labels():
    mod = codec()
    rng = np.random.default_rng(192)
    source = episode(count=120, starts=range(111))
    lower = np.asarray(action_contract.LOWER)
    upper = np.asarray(action_contract.UPPER)
    source['state'][:, :7] = rng.uniform(lower+.0001, upper-.0001, (120, 7))
    source['actions'][:] = rng.uniform(lower+.0001, upper-.0001, (120, 7))
    stats = mod.fit_stats([source])
    for start in source['starts']:
        absolute = source['actions'][start:start+10].copy()
        anchor = source['state'][start, :7]
        result = mod.decode_chunk(mod.encode_chunk(absolute, anchor, 'CONTACT', stats), anchor, 'CONTACT', stats)
        np.testing.assert_allclose(result, absolute, atol=2e-6, rtol=0)


def test_rejects_unknown_phase_and_changed_source_binding():
    mod = codec()
    source = episode()
    stats = mod.fit_stats([source])
    with pytest.raises(ValueError, match='phase'):
        mod.encode_chunk(source['actions'][:10], source['state'][0, :7], 'FREE', stats)
    stats['action_encoding_binding'] = {'version': 'other'}
    with pytest.raises(ValueError, match='binding'):
        mod.decode_chunk(np.zeros((10, 7)), source['state'][0, :7], 'CONTACT', stats)
