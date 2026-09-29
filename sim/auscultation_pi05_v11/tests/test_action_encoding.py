"""CPU codec tests; small artificial TRAIN windows and approved plan metadata."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'sim/auscultation_pi05_v11/action_encoding.py'


def test_declared_codec_is_implemented():
    assert SOURCE.is_file(), 'The declared new codec must exist before callers can use it'


@pytest.fixture(scope='module')
def codec():
    assert SOURCE.is_file(), 'New target interval codec implementation is missing'
    spec = importlib.util.spec_from_file_location('v11_codec_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def bounds(codec):
    return codec.build_bounds_contract(ROOT)


@pytest.fixture
def episodes(bounds):
    result = []
    for k, phase in enumerate(('FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT')):
        limits = bounds['targets']['R1']
        low, high = np.asarray(limits['lower_rad']), np.asarray(limits['upper_rad'])
        actions = np.tile((low + high) / 2, (15, 1))
        actions += np.arange(15)[:, None] * (np.arange(7)[None, :] + 1) * .0001
        if phase == 'FREE':
            actions[:, 3] = 1.7000253365040408e-6 + np.arange(15)**2 * .0001
        state = np.zeros((15, 41), np.float32)
        state[:, :7] = actions + np.arange(15)[:, None] * .0003
        state[0, 3] = 1.0497749826754443e-5 if phase == 'FREE' else state[0, 3]
        result.append(dict(episode_id='synthetic_train_' + phase, split='train',
                           actions=actions.astype(np.float32), state=state,
                           phases=np.full(15, phase), targets=np.full(15, 'R1'), starts=[0, 3, 5]))
    return result


@pytest.fixture
def stats(codec, bounds, episodes):
    return codec.fit_stats(episodes, bounds)


def reseal(codec, obj, field):
    obj[field] = codec.digest({k: v for k, v in obj.items() if k != field})


def test_source_bound_table_and_exact_inward_endpoints(codec, bounds):
    assert codec.validate_bounds_contract(bounds) == bounds['bounds_sha256']
    assert len(bounds['source_sha256']) == 12
    assert set(bounds['targets']) == {'R1','L1','R2','L2','R3','L3','R4','L4','R5','L5'}
    for target, values in bounds['targets'].items():
        assert values['target'] == target
        lo, hi = np.asarray(values['lower_rad']), np.asarray(values['upper_rad'])
        original_lo, original_hi = np.asarray(values['original_lower_rad']), np.asarray(values['original_upper_rad'])
        assert np.array_equal(lo, lo.astype(np.float32).astype(np.float64))
        assert np.array_equal(hi, hi.astype(np.float32).astype(np.float64))
        assert np.all(lo >= original_lo) and np.all(hi <= original_hi)
        assert np.all(np.nextafter(lo.astype(np.float32), np.float32(-np.inf)).astype(np.float64) < original_lo)
        assert np.all(np.nextafter(hi.astype(np.float32), np.float32(np.inf)).astype(np.float64) > original_hi)


def test_builder_rejects_changed_actual_source_before_parsing(codec, bounds, tmp_path):
    for rel in bounds['source_sha256']:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    chosen = tmp_path / 'sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_R1.json'
    chosen.write_text('{not-json')
    with pytest.raises(ValueError, match='source|hash'):
        codec.build_bounds_contract(tmp_path)


@pytest.mark.parametrize('change', ['source', 'target', 'original', 'endpoint', 'mode'])
def test_resealed_bounds_cannot_change_approved_projection(codec, bounds, change):
    bad = copy.deepcopy(bounds)
    if change == 'source': bad['source_sha256'][next(iter(bad['source_sha256']))] = '0'*64
    elif change == 'target': bad['targets']['R1']['target'] = 'L1'
    elif change == 'original': bad['targets']['R1']['original_lower_rad'][0] += .001
    elif change == 'endpoint': bad['targets']['R1']['lower_rad'][0] += .001
    else: bad['mode'] = 'B_horizon_free_approach'
    reseal(codec, bad, 'bounds_sha256')
    with pytest.raises(ValueError): codec.validate_bounds_contract(bad)


def test_empty_narrow_interval_rejected_without_clamp(codec):
    lo = np.full(7, 1.0 + 1e-9)
    hi = np.full(7, 1.0 + 2e-9)
    with pytest.raises(ValueError, match='empty|interval'): codec.inward_endpoints(lo, hi)
    lo[0] = np.nan
    with pytest.raises(ValueError, match='finite'): codec.inward_endpoints(lo, hi)


def test_all70_inverse_roundtrip_and_source_dtype(codec, bounds, episodes):
    for ep in episodes:
        q = ep['actions'][:10]
        x = codec.inverse_coordinates(q, 'R1', bounds)
        assert x.dtype == np.float64 and x.shape == (10, 7)
        out = codec.forward_coordinates(x, 'R1', bounds)
        assert out.dtype == np.float32 and out.shape == (10, 7)
        assert np.max(np.abs(out.astype(np.float64) - q)) <= 2e-6


@pytest.mark.parametrize('value', ['lower', 'upper', 'outside', 'nan', 'inf'])
def test_inverse_rejects_bad_label_at_any_horizon(codec, bounds, episodes, value):
    q = episodes[0]['actions'][:10].astype(np.float64)
    v = bounds['targets']['R1']
    q[9, 6] = {'lower':v['lower_rad'][6], 'upper':v['upper_rad'][6], 'outside':v['upper_rad'][6]+.1, 'nan':np.nan, 'inf':np.inf}[value]
    with pytest.raises(ValueError): codec.inverse_coordinates(q, 'R1', bounds)


def test_monotone_saturating_not_strict_float_bijection(codec, bounds):
    v = bounds['targets']['R1'];lo=np.asarray(v['lower_rad']);hi=np.asarray(v['upper_rad'])
    previous = None; equal = 0
    for t in np.r_[-1e30, np.linspace(-.2, 1.2, 121), 1e30]:
        x = np.tile(lo + t * (hi-lo), (10, 1))
        out = codec.forward_coordinates(x, 'R1', bounds)
        assert np.all(out >= lo) and np.all(out <= hi)
        if previous is not None:
            assert np.all(out >= previous)
            equal += int((out == previous).sum())
        previous = out
    assert equal > 0


def test_stats_fit_uses_exact_B_axes_and_each_explicit_window_once(codec, bounds, episodes, stats):
    assert stats['fit_window_count'] == 15
    assert stats['fit_episode_ids'] == [e['episode_id'] for e in episodes]
    for ep in episodes:
        phase=ep['phases'][0];values=np.stack([codec.inverse_coordinates(ep['actions'][s:s+10], 'R1', bounds)-ep['state'][s,:7].astype(np.float64) for s in ep['starts']])
        axes=0 if phase in ('FREE','APPROACH') else (0,1)
        qlo,qhi=np.quantile(values,[.01,.99],axis=axes,method='linear')
        center=(qlo+qhi)/2;span=np.maximum(qhi-qlo,.002);own=stats['actions_by_phase'][phase]
        assert np.array_equal(own['low'],center-span/2)
        assert np.array_equal(own['high'],center+span/2)
        assert np.asarray(own['low']).shape == ((10,7) if phase in ('FREE','APPROACH') else (7,))
        assert own['window_count']==3 and own['residual_count']==30
        assert own['fit_axis'] == ('window_only_separate_horizon_joint' if phase in ('FREE','APPROACH') else 'window_and_horizon_per_joint')


def test_serialization_hash_roundtrip_and_saved_stats_only(codec, bounds, episodes, stats, monkeypatch):
    restored=json.loads(json.dumps(stats,allow_nan=False))
    h=codec.validate_stats(restored,expected_sha256=stats['normalization_sha256'],expected_bounds_sha256=bounds['bounds_sha256'])
    assert h==stats['normalization_sha256']
    monkeypatch.setattr(codec,'fit_stats',lambda *a,**kw:pytest.fail('Encode/decode must not refit'))
    for ep in episodes:
        for start in ep['starts']:
            q=ep['actions'][start:start+10];a=ep['state'][start,:7];phase=str(ep['phases'][start])
            z=codec.encode_chunk(q,a,phase,'R1',restored)
            decoded=codec.decode_chunk(z,a,phase,'R1',restored)
            assert z.shape==(10,7) and z.dtype==np.float32 and decoded.dtype==np.float32
            assert np.max(np.abs(decoded.astype(np.float64)-q)) <= 2e-6
            latent=codec.denormalize_chunk(z,phase,'R1',restored)
            assert latent.dtype==np.float64
            assert np.array_equal(decoded,codec.forward_coordinates(latent+a.astype(np.float64),'R1',bounds))


def test_anchor_is_one_original_measurement_for_full_horizon(codec, episodes, stats):
    ep=episodes[0];q=ep['actions'][:10];a=ep['state'][0,:7];z=codec.encode_chunk(q,a,'FREE','R1',stats)
    changed=a.copy();changed[0]+=.01
    out=codec.decode_chunk(z,changed,'FREE','R1',stats)
    assert np.all(out[:,0] > q[:,0]+.009)
    with pytest.raises(ValueError): codec.decode_chunk(z,ep['state'][:10,:7],'FREE','R1',stats)


def test_zero_latent_residual_does_not_promise_hold(codec, bounds):
    a=np.array([0.,0.,0.,1.0497749826754443e-5,0.,0.,0.])
    out=codec.forward_coordinates(np.tile(a,(10,1)),'R1',bounds)
    assert .00069 < out[0,3] < .00071 and out[0,3] != a[3]
    assert codec.action_encoding_binding(bounds)['zero_residual_is_hold'] is False


@pytest.mark.parametrize('z_value', [-np.finfo(np.float32).max, -100., 0., 100., np.finfo(np.float32).max])
def test_finite_float32_extremes_decode_all70_inside_original(codec,bounds,episodes,stats,z_value):
    z=np.full((10,7),z_value,np.float32)
    limits=bounds['targets']['R1']
    for ep in episodes:
        out=codec.decode_chunk(z,ep['state'][0,:7],str(ep['phases'][0]),'R1',stats)
        assert np.isfinite(out).all()
        assert np.all(out>=limits['original_lower_rad']) and np.all(out<=limits['original_upper_rad'])


@pytest.mark.parametrize('bad_z', [np.full((10,7),np.nan),np.full((10,7),np.inf),np.full((10,7),20.00000001),np.ones((9,7)),np.ones((10,7),bool)])
def test_decode_rejects_nonfinite_or_non_float32_exact_or_shape(codec,episodes,stats,bad_z):
    with pytest.raises(ValueError): codec.decode_chunk(bad_z,episodes[0]['state'][0,:7],'FREE','R1',stats)


@pytest.mark.parametrize('change', ['mode','source','range','bounds','axis','hash','count'])
def test_wrong_or_resealed_stats_rejected(codec,stats,change):
    bad=copy.deepcopy(stats)
    if change=='mode':bad['mode']='B_horizon_free_approach'
    elif change=='source':bad['action_encoding_binding']['source_sha256']='0'*64
    elif change=='range':bad['actions_by_phase']['FREE']['high'][9][6]=bad['actions_by_phase']['FREE']['low'][9][6]
    elif change=='bounds':bad['bounds_contract']['targets']['R1']['upper_rad'][0]+=.1
    elif change=='axis':bad['actions_by_phase']['CONTACT']['fit_axis']='window_only_separate_horizon_joint'
    elif change=='count':bad['fit_window_count']+=1
    else:bad['normalization_sha256']='0'*64
    if change!='hash':reseal(codec,bad,'normalization_sha256')
    with pytest.raises(ValueError):codec.validate_stats(bad)


@pytest.mark.parametrize('target', ['R0','r1','R1 ',None,1])
def test_explicit_canonical_target_required(codec,episodes,stats,target):
    with pytest.raises(ValueError):codec.decode_chunk(np.zeros((10,7),np.float32),episodes[0]['state'][0,:7],'FREE',target,stats)


@pytest.mark.parametrize('change', ['test','duplicate_id','duplicate_start','bool_start','incomplete','cross_phase','cross_target','unknown_target','nan_anchor','bad_state'])
def test_fit_rejects_unqualified_numeric_view(codec,bounds,episodes,change):
    bad=copy.deepcopy(episodes)
    if change=='test':bad[0]['split']='test'
    elif change=='duplicate_id':bad[1]['episode_id']=bad[0]['episode_id']
    elif change=='duplicate_start':bad[0]['starts']=[0,0]
    elif change=='bool_start':bad[0]['starts']=[True]
    elif change=='incomplete':bad[0]['starts']=[6]
    elif change=='cross_phase':bad[0]['phases'][9]='HOLD'
    elif change=='cross_target':bad[0]['targets'][9]='L1'
    elif change=='unknown_target':bad[0]['targets'][9]='XX'
    elif change=='nan_anchor':bad[0]['state'][0,0]=np.nan
    else:bad[0]['state']=bad[0]['state'][:,:7]
    with pytest.raises(ValueError):codec.fit_stats(bad,bounds)


def test_no_mutation_and_new_identity(codec,bounds,episodes):
    old_bounds=copy.deepcopy(bounds);old_episodes=copy.deepcopy(episodes)
    stats=codec.fit_stats(episodes,bounds)
    assert bounds==old_bounds
    for a,b in zip(episodes,old_episodes):
        for key in ('actions','state','phases','targets'):assert np.array_equal(a[key],b[key])
    b=codec.action_encoding_binding(bounds)
    assert b['bounded_by_codec'] is True and b['action_clipping'] is False and b['posthoc_projection'] is False
    assert b['tau_rad']==.001 and b['version']=='wam_pi05_v11_target_interval_latent_residual_v1'
    assert stats['mode']==codec.MODE and stats['schema'] != 'wam_paired_fixed_observation_horizon_normalization_v1'
    with pytest.raises(ValueError):codec.validate_stats(stats,expected_sha256='0'*64)
    with pytest.raises(ValueError):codec.validate_stats(stats,expected_bounds_sha256='0'*64)


def test_fit_records_all_window_production_axis_roundtrip(codec,episodes,stats):
    errors=[]
    for ep in episodes:
        for s in ep['starts']:
            q=ep['actions'][s:s+10];a=ep['state'][s,:7];phase=str(ep['phases'][s])
            z=codec.encode_chunk(q,a,phase,'R1',stats)
            errors.append(float(np.abs(codec.decode_chunk(z,a,phase,'R1',stats).astype(np.float64)-q).max()))
    assert stats['fit_roundtrip_max_error_rad']==max(errors)
    assert stats['fit_roundtrip_tolerance_rad']==2e-6


def test_fit_rejects_finite_anchor_cancellation_that_loses_original_tolerance(codec,bounds,episodes):
    for ep in episodes:
        ep['state']=ep['state'].astype(np.float64)
        ep['state'][:,:7]=1e12
    with pytest.raises(ValueError,match='roundtrip'):
        codec.fit_stats(episodes,bounds)
