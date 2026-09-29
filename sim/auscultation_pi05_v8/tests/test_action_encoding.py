"""Synthetic numeric fixtures only; never acquisition/qualification evidence."""
import copy
import importlib
import importlib.util
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import action_encoding as OLD

MODULE='sim.auscultation_pi05_v8.action_encoding'
PHASES=('FREE','APPROACH','CONTACT','HOLD','RETRACT')
A='A_pooled_control';B='B_horizon_free_approach'

@pytest.fixture
def codec():
    if importlib.util.find_spec(MODULE) is None:
        from types import SimpleNamespace
        return SimpleNamespace(fit_stats=lambda *args: pytest.fail('New paired codec is not implemented'))
    return importlib.import_module(MODULE)

def episodes():
    result=[]
    for number in range(2):
        n=100;t=np.arange(n,dtype=np.float64)
        state=np.zeros((n,41),np.float32)
        state[:,:7]=(t[:,None]*.013+np.arange(7)*.003+number*.02).astype(np.float32)
        actions=(state[:,:7].astype(float)+.009+.0002*t[:,None]**1.1*np.arange(1,8)).astype(np.float32)
        result.append(dict(episode_id='synthetic_'+str(number),split='train',actions=actions,state=state,
            phases=np.repeat(PHASES,20),starts=[p*20+i for p in range(5) for i in (0,2,5,10)]))
    return result

def digest_arrays(records):
    return [(e['actions'].tobytes(),e['state'].tobytes(),e['phases'].tobytes(),list(e['starts'])) for e in records]

def test_control_is_bitwise_parent_and_no_input_mutation(codec):
    records=episodes();before=digest_arrays(records);pooled=OLD.fit_stats(records)
    stats=codec.fit_stats(records,A)
    assert stats['parent_pooled_stats']==pooled
    assert stats['actions_by_phase']==pooled['actions_by_phase']
    for e in records:
        for start in e['starts']:
            phase=str(e['phases'][start]);chunk=e['actions'][start:start+10];q=e['state'][start,:7]
            x=codec.encode_chunk(chunk,q,phase,stats,mode=A)
            assert x.tobytes()==OLD.encode_chunk(chunk,q,phase,pooled).tobytes()
            assert codec.decode_chunk(x,q,phase,stats,mode=A).tobytes()==OLD.decode_chunk(x,q,phase,pooled).tobytes()
    assert before==digest_arrays(records)

def test_horizon_quantiles_each_window_once_fixed_anchor(codec):
    records=episodes();stats=codec.fit_stats(records,B)
    for phase in ('FREE','APPROACH'):
        x=np.stack([e['actions'][s:s+10].astype(float)-e['state'][s,:7].astype(float)
                    for e in records for s in e['starts'] if e['phases'][s]==phase])
        lo,hi=np.quantile(x,[.01,.99],axis=0,method='linear');center=(lo+hi)/2;span=np.maximum(hi-lo,.002)
        own=stats['actions_by_phase'][phase]
        np.testing.assert_array_equal(own['low'],center-span/2)
        np.testing.assert_array_equal(own['high'],center+span/2)
        assert np.asarray(own['low']).shape==(10,7)
        assert own['window_count']==len(x) and own['residual_count']==len(x)*10
        assert own['fit_axis']=='window_only_separate_horizon_joint'
        assert not np.array_equal(np.asarray(own['low'])[0],np.asarray(own['low'])[9])

def test_untouched_phases_exact_parent_stats_and_math(codec):
    records=episodes();stats=codec.fit_stats(records,B);old=OLD.fit_stats(records)
    for phase in ('CONTACT','HOLD','RETRACT'):
        assert stats['actions_by_phase'][phase]==old['actions_by_phase'][phase]
        q=records[0]['state'][0,:7];x=np.linspace(-8,12,70,dtype=np.float32).reshape(10,7)
        y=codec.decode_chunk(x,q,phase,stats,mode=B)
        assert y.tobytes()==OLD.decode_chunk(x,q,phase,old).tobytes()
        assert codec.encode_chunk(y,q,phase,stats,mode=B).tobytes()==OLD.encode_chunk(y,q,phase,old).tobytes()

def test_all_roundtrips_and_no_clipping_original_gate(codec):
    from sim.auscultation_pi05_v6.action_contract import bounded,UPPER
    for mode in (A,B):
        records=episodes();stats=codec.fit_stats(records,mode)
        for e in records:
            for start in e['starts']:
                q=e['state'][start,:7];phase=str(e['phases'][start]);original=e['actions'][start:start+10]
                x=codec.encode_chunk(original,q,phase,stats,mode=mode)
                restored=codec.decode_chunk(x,q,phase,stats,mode=mode)
                assert x.dtype==restored.dtype==np.float32
                assert np.max(abs(restored.astype(float)-original))<=2e-6
        raw=np.repeat((np.asarray(UPPER)+1)[None],10,axis=0);q=np.zeros(7)
        x=codec.encode_chunk(raw,q,'FREE',stats,mode=mode)
        assert np.any(abs(x)>1)
        restored=codec.decode_chunk(x,q,'FREE',stats,mode=mode)
        np.testing.assert_allclose(restored,raw,atol=2e-6,rtol=0)
        assert not bounded(restored,'FREE')

def test_engineering_floor_separate_each_cell(codec):
    e=episodes()[0];e['actions'][:]=.001;e['state'][:]=0
    stats=codec.fit_stats([e],B)
    own=stats['actions_by_phase']['FREE']
    np.testing.assert_allclose(np.asarray(own['high'])-own['low'],.002,rtol=0,atol=1e-18)
    np.testing.assert_allclose((np.asarray(own['high'])+own['low'])/2,float(np.float32(.001)),rtol=0,atol=1e-18)

@pytest.mark.parametrize('mutation',[lambda r:r[0].update(split='validation'),lambda r:r[0].update(split='test'),
    lambda r:r[1].update(episode_id=r[0]['episode_id']),lambda r:r[0]['starts'].append(r[0]['starts'][0]),
    lambda r:r[0]['starts'].append(19),lambda r:r[0]['starts'].append(True),
    lambda r:r[0].update(starts=[]),lambda r:r[0]['state'].__setitem__((0,0),np.nan),
    lambda r:r[0]['actions'].__setitem__((0,0),np.inf),lambda r:r[0].update(actions=np.zeros((100,6))),
    lambda r:r[0].update(state=np.full((100,41),'1')),lambda r:r[0].update(phases=np.repeat('UNKNOWN',100))])
def test_fit_rejects_invalid_records(codec,mutation):
    records=episodes();mutation(records)
    for mode in (A,B):
        with pytest.raises(ValueError):codec.fit_stats(records,mode)

def test_gaps_do_not_generate_omitted_starts(codec):
    records=episodes();records[0]['starts']=[0,10];records[1]['starts']=[20]
    stats=codec.fit_stats(records,B)
    assert stats['actions_by_phase']['FREE']['window_count']==2
    assert stats['actions_by_phase']['APPROACH']['window_count']==1
    assert stats['fit_window_count']==3

def test_new_mode_source_norm_integrity_rejects_legacy_mismatch(codec):
    records=episodes();a=codec.fit_stats(records,A);b=codec.fit_stats(records,B);q=np.zeros(7);x=np.zeros((10,7))
    assert codec.normalization_digest(a,mode=A)!=codec.normalization_digest(b,mode=B)
    assert codec.action_encoding_binding(A)!=codec.action_encoding_binding(B)
    for stats,mode in ((OLD.fit_stats(records),A),(a,B),(b,A)):
        with pytest.raises(ValueError):codec.decode_chunk(x,q,'FREE',stats,mode=mode)
    bad=copy.deepcopy(b);bad['actions_by_phase']['FREE']['low'][0][0]+=.1
    with pytest.raises(ValueError):codec.decode_chunk(x,q,'FREE',bad,mode=B)
    with pytest.raises(ValueError):codec.validate_stats(b,mode=B,expected_sha256=codec.normalization_digest(a,mode=A))
    with pytest.raises(ValueError):codec.fit_stats(records,'old')

@pytest.mark.parametrize('bad',[np.zeros((7,10)),np.full((10,7),np.nan),np.full((10,7),'1')])
def test_encode_decode_reject_malformed_chunks(codec,bad):
    s=codec.fit_stats(episodes(),B)
    for fn in (codec.encode_chunk,codec.decode_chunk):
        with pytest.raises(ValueError):fn(bad,np.zeros(7),'FREE',s,mode=B)
