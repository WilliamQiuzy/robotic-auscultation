"""Pure synthetic reference views; canonical qualification is tested separately."""
import copy
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import action_encoding as OLD
from sim.auscultation_pi05_v8 import action_encoding as B, training_data as BD
from test_action_encoding import episodes, SIX

@pytest.fixture
def data():
    try: return importlib.import_module('sim.auscultation_pi05_v9.training_data')
    except ModuleNotFoundError:
        def absent(*a,**kw): pytest.fail('Conditional dataset view is not implemented')
        return SimpleNamespace(view_parent=absent,load_dataset=absent)

def make_parent(monkeypatch):
    calls=[]
    def observation(rgb,state,phase,target,tokenizer,norm):
        calls.append((rgb,state,phase,target,tokenizer,norm))
        return {'state':state.copy(),'rgb':rgb.copy(),'tokens':np.array([len(phase),len(target)],np.int32)}
    api=SimpleNamespace(core_sources=lambda:dict(BD.EXPECTED_PARENT_SOURCES),observation=observation,
        stack_observations=lambda rows:{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    monkeypatch.setattr(BD,'_parent_module',lambda:api)
    records=episodes(); indices={k:[] for k in ('train','validation','test')}
    for i,e in enumerate(records):
        e['targets']=np.repeat('R4',100);e['times']=np.arange(100)*.05
        indices['train'].extend((i,s) for s in e['starts'])
    held=copy.deepcopy(records[0]);held.update(split='validation',episode_id='heldout');records.append(held)
    indices['validation']=[(2,0)]
    rgb=np.arange(100*2*2*3,dtype=np.uint8).reshape(100,2,2,3)
    base=SimpleNamespace(episodes=records,indices=indices,dynamics={'test':True},manifest={'test':True},
        data_hash='synthetic',normalization={'state':{'identity':True},'encoding_binding':{'identity':True},
        'residual_actions':OLD.fit_stats(records[:2]),'action_encoding_binding':OLD.action_encoding_binding()},
        parent=SimpleNamespace(rgb=lambda i:rgb))
    return BD.view_parent(base,B.B),calls

def test_preserves_B_references_observations_and_new_identity(data,monkeypatch):
    b,calls=make_parent(monkeypatch);c=data.view_parent(b)
    assert c.parent is b and c.indices is b.indices and c.episodes is b.episodes
    assert c.manifest is b.manifest and c.dynamics is b.dynamics and c.rgb(0) is b.rgb(0)
    assert c.normalization['state'] is b.normalization['state']
    assert c.normalization['encoding_binding'] is b.normalization['encoding_binding']
    assert c.data_hash != b.data_hash and c.mode=='C_rational_q4'
    assert c.normalization['residual_actions']['fit_episode_ids']==['synthetic_0','synthetic_1']
    obs,actions=data.make_batch(c,'train',[0,1],None);bo,ba=BD.make_batch(b,'train',[0,1],None)
    for key in obs: assert obs[key].tobytes()==bo[key].tobytes()
    assert actions[:,:,SIX].tobytes()==ba[:,:,SIX].tobytes()
    assert not np.array_equal(actions[:,:,3],ba[:,:,3])
    for i,(ei,s) in enumerate(c.indices['train'][:2]):
        e=c.episodes[ei];raw=data.decode_prediction(c,actions[i],e['state'][s],str(e['phases'][s]))
        assert np.max(abs(raw.astype(float)-e['actions'][s:s+10]))<=2e-6
    data.validate_binding(c,data.dataset_binding(c))
    with pytest.raises(ValueError):data.validate_binding(c,BD.dataset_binding(b))

def test_saved_normalization_loading_uses_no_C_fit(data,monkeypatch):
    b,_=make_parent(monkeypatch);c=data.view_parent(b)
    norm=copy.deepcopy(c.normalization);digest=data._digest(norm)
    monkeypatch.setattr(data.codec,'fit_stats',lambda *a,**kw:pytest.fail('Unexpected C refit'))
    again=data.view_parent(b,normalization=norm,expected_normalization_sha256=digest)
    assert again.data_hash==c.data_hash and data.dataset_binding(again)==data.dataset_binding(c)
    data.make_batch(again,'train',[0],None)
    with pytest.raises(ValueError): data.view_parent(b,normalization=norm,expected_normalization_sha256='0'*64)
    with pytest.raises(ValueError): data.view_parent(b,normalization=b.normalization)

@pytest.mark.parametrize('change', ['mode','source','state','stats','indices'])
def test_mutated_view_rejected(data,monkeypatch,change):
    b,_=make_parent(monkeypatch);c=data.view_parent(b)
    if change=='mode': object.__setattr__(c,'mode',B.B)
    elif change=='source': c.source_sha256['v9/action_encoding.py']='0'*64
    elif change=='state': c.normalization['state']={'changed':True}
    elif change=='stats': c.normalization['residual_actions']['actions_by_phase']['FREE']['low'][0][3]+=.1
    else: b.indices['train'].append(b.indices['train'][0])
    with pytest.raises(ValueError): data.dataset_binding(c)

def test_bad_mode_manifest_and_batch_rejected(data,monkeypatch,tmp_path):
    b,_=make_parent(monkeypatch);c=data.view_parent(b)
    with pytest.raises(ValueError):data.view_parent(b,mode=B.B)
    with pytest.raises(ValueError):data.view_parent(b.parent)
    p=tmp_path/'manifest.json';p.write_text('{}')
    with pytest.raises(ValueError):data.load_dataset(p)
    for split,indices in [('no',[0]),('train',[]),('train',[True]),('train',[-1]),('train',[10000])]:
        with pytest.raises(ValueError):data.make_batch(c,split,indices,None)
