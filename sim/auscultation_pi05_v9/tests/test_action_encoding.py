"""Conditional codec tests; synthetic arrays are not qualification evidence."""
import copy
import importlib
import numpy as np
import pytest
from sim.auscultation_pi05_v8 import action_encoding as B

MODE = 'C_rational_q4'
PHASES = ('FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT')
SIX = [0, 1, 2, 4, 5, 6]

@pytest.fixture
def codec():
    try:
        return importlib.import_module('sim.auscultation_pi05_v9.action_encoding')
    except ModuleNotFoundError:
        from types import SimpleNamespace
        def absent(*args, **kwargs): pytest.fail('Conditional rational codec is not implemented')
        return SimpleNamespace(fit_stats=absent, inverse_q4=absent, decode_q4=absent, U_SAFE_RAD=2.4434609413146973)

def episodes():
    records = []
    for j in range(2):
        t = np.arange(100)
        state = np.zeros((100, 41), np.float32)
        state[:, :7] = .1 + t[:, None] * .003 + np.arange(7) * .004 + j * .02
        actions = (state[:, :7].astype(float) + .001 * np.sin(t[:, None] + np.arange(7))).astype(np.float32)
        state[0, 3] = 1.05e-5
        actions[0, 3] = 2e-6
        records.append(dict(episode_id=f'synthetic_{j}', split='train', state=state, actions=actions,
                            phases=np.repeat(PHASES, 20), starts=[p*20+i for p in range(5) for i in (0,2,5,10)]))
    return records

def test_inverse_and_decode_known_signed_values(codec):
    a = np.array([1., 1., 1.])
    q = np.array([.5, 1., 1.5])
    expected = np.array([-1., 0., (codec.U_SAFE_RAD-1.)*.5/(codec.U_SAFE_RAD-1.5)])
    np.testing.assert_array_equal(codec.inverse_q4(q, a), expected)
    np.testing.assert_array_equal(codec.decode_q4(expected, a), q.astype(np.float32))

@pytest.mark.parametrize('anchor', [0., 1.05e-5, .4, 2.4434609413146973])
def test_monotone_finite_extremes_and_zero_retention(codec, anchor):
    r = np.array([-np.finfo(float).max, -1e20, -1., -1e-6, 0., 1e-6, 1., 1e20, np.finfo(float).max])
    out = codec.decode_q4(r, anchor)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert np.all(np.diff(out.astype(float)) >= 0) and np.all(out >= 0) and np.all(out.astype(float) <= codec.U_SAFE_RAD)
    assert out[4].tobytes() == np.float32(anchor).tobytes()

@pytest.mark.parametrize('a,q', [(1.,0.),(1.,2.4434609413146973),(-1e-45,1.),(3.,1.),(1.,-1e-45),(1.,3.),(np.nan,1.),(1.,np.inf),(1.,np.nan)])
def test_inverse_rejects_domains_and_nonanchor_endpoints(codec, a, q):
    with pytest.raises(ValueError): codec.inverse_q4(q, a)

def test_equal_endpoint_degeneracies_and_unrepresentable_inverse(codec):
    for a in (0., codec.U_SAFE_RAD):
        assert codec.inverse_q4(a, a) == 0
        assert codec.decode_q4(0., a) == np.float32(a)
    assert codec.decode_q4(-1., 0.) == 0
    assert codec.decode_q4(1., codec.U_SAFE_RAD) == np.float32(codec.U_SAFE_RAD)
    with pytest.raises(ValueError): codec.inverse_q4(np.nextafter(np.float32(0),np.float32(1)), 1.)
    for r in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError): codec.decode_q4(r, .5)

def test_fit_axes_floor_parent_statistics_and_roundtrip(codec):
    records = episodes(); original = copy.deepcopy(records)
    parent = B.fit_stats(records, B.B); stats = codec.fit_stats(records, MODE, parent_stats=parent)
    assert stats['parent_B_stats'] == parent
    assert stats['fit_windows_sha256'] == parent['fit_windows_sha256']
    for phase in PHASES:
        values = np.stack([codec.inverse_q4(e['actions'][s:s+10,3],e['state'][s,3]) for e in records for s in e['starts'] if e['phases'][s] == phase])
        lo, hi = np.quantile(values, [.01,.99], axis=0 if phase in ('FREE','APPROACH') else None, method='linear')
        center=(lo+hi)/2; span=np.maximum(hi-lo,.002)
        own=stats['actions_by_phase'][phase]; old=parent['actions_by_phase'][phase]
        for key in ('low','high','observed_min','observed_max'):
            assert np.asarray(own[key])[...,SIX].tobytes() == np.asarray(old[key])[...,SIX].tobytes()
        np.testing.assert_array_equal(np.asarray(own['low'])[...,3], center-span/2)
        np.testing.assert_array_equal(np.asarray(own['high'])[...,3], center+span/2)
    for e in records:
        for s in e['starts']:
            q=e['state'][s,:7]; raw=e['actions'][s:s+10]; phase=str(e['phases'][s])
            x=codec.encode_chunk(raw,q,phase,stats,mode=MODE); y=codec.decode_chunk(x,q,phase,stats,mode=MODE)
            bx=B.encode_chunk(raw,q,phase,parent,mode=B.B); by=B.decode_chunk(bx,q,phase,parent,mode=B.B)
            assert x[:,SIX].tobytes()==bx[:,SIX].tobytes() and y[:,SIX].tobytes()==by[:,SIX].tobytes()
            assert x.dtype == y.dtype == np.float32 and abs(y.astype(float)-raw).max() <= 2e-6
    for old,new in zip(original,records):
        for key in ('state','actions','phases'): assert old[key].tobytes()==new[key].tobytes()

def test_floor_and_out_of_quantile_not_clipped(codec):
    records=episodes()
    for e in records: e['actions'][:,3]=.2; e['state'][:,3]=.2
    stats=codec.fit_stats(records)
    for p in PHASES:
        own=stats['actions_by_phase'][p]
        np.testing.assert_array_equal(np.asarray(own['high'])[...,3]-np.asarray(own['low'])[...,3],.002)
    raw=np.zeros((10,7),np.float32);raw[:]=.2;raw[:,0]=20.;raw[:,3]=.9;q=np.full(7,.2,np.float32)
    z=codec.encode_chunk(raw,q,'FREE',stats)
    assert np.any(abs(z)>1)
    decoded=codec.decode_chunk(z,q,'FREE',stats)
    assert decoded[0,0] > 19 and decoded[0,3] > .8

def test_saved_stats_reload_never_refits_and_rejects_mode_source_or_parent_tamper(codec,monkeypatch):
    stats=codec.fit_stats(episodes()); q=np.full(7,.2,np.float32); x=np.zeros((10,7),np.float32)
    monkeypatch.setattr(codec,'fit_stats',lambda *a,**k:pytest.fail('Deployment refit'))
    codec.decode_chunk(x,q,'FREE',stats)
    for change in ('mode','source','six','range'):
        bad=copy.deepcopy(stats)
        if change=='mode': bad['mode']=B.B
        elif change=='source': bad['action_encoding_binding']['source_sha256']='0'*64
        elif change=='six': bad['actions_by_phase']['FREE']['low'][0][0]+=.1
        else: bad['actions_by_phase']['FREE']['high'][0][3]=bad['actions_by_phase']['FREE']['low'][0][3]
        bad['normalization_sha256']=codec._digest({k:v for k,v in bad.items() if k!='normalization_sha256'})
        with pytest.raises(ValueError): codec.decode_chunk(x,q,'FREE',bad)
    with pytest.raises(ValueError): codec.validate_stats(stats,expected_sha256='0'*64)
    with pytest.raises(ValueError): codec.decode_chunk(x,q,'FREE',stats,mode=B.B)
    with pytest.raises(ValueError): codec.decode_chunk(x,q,'FREE',stats['parent_B_stats'])

@pytest.mark.parametrize('mutation',[lambda r:r[0].update(split='test'),lambda r:r[0]['starts'].append(19),lambda r:r[0]['starts'].append(True),lambda r:r[0]['starts'].append(0),lambda r:r[0]['state'].__setitem__((99,3),-1e-6),lambda r:r[0]['actions'].__setitem__((0,3),0.)])
def test_fit_invalid_or_unsupported_records_rejected(codec,mutation):
    records=episodes();mutation(records)
    with pytest.raises(ValueError): codec.fit_stats(records)

@pytest.mark.parametrize('bad',[np.zeros((7,10)),np.full((10,7),np.nan),np.full((10,7),'1')])
def test_chunks_fail_closed(codec,bad):
    stats=codec.fit_stats(episodes())
    for fn in (codec.encode_chunk,codec.decode_chunk):
        with pytest.raises(ValueError):fn(bad,np.full(7,.2,np.float32),'FREE',stats)
    with pytest.raises(ValueError):codec.decode_chunk(np.zeros((10,7)),np.array([0,0,0,-1e-8,0,0,0]),'FREE',stats)

def test_extreme_serialized_q4_does_not_overflow_irrelevant_B_math(codec):
    stats=codec.fit_stats(episodes());q=np.full(7,.2,np.float32)
    for r in (-np.finfo(np.float32).max,np.finfo(np.float32).max):
        z=np.zeros((10,7),np.float32);z[:,3]=r
        decoded=codec.decode_chunk(z,q,'FREE',stats)
        assert np.isfinite(decoded).all() and np.all(decoded[:,3]>=0) and np.all(decoded[:,3].astype(float)<=codec.U_SAFE_RAD)
    # A malformed wider-than-float32 normalized payload is not a valid model sample.
    z=np.zeros((10,7),np.float64);z[:,3]=np.finfo(np.float64).max
    with pytest.raises(ValueError):codec.decode_chunk(z,q,'FREE',stats)
