"""Synthetic parent references exercise view logic, not qualification evidence."""
import copy,importlib,importlib.util
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v7 import action_encoding as OLD
from test_action_encoding import episodes,A,B

@pytest.fixture
def data():
    name='sim.auscultation_pi05_v8.training_data'
    if importlib.util.find_spec(name) is None:
        return SimpleNamespace(view_parent=lambda *args:pytest.fail('New qualified dataset view is not implemented'))
    return importlib.import_module(name)

def parent_fixture():
    records=episodes();indices={s:[] for s in ('train','validation','test')}
    for i,e in enumerate(records):
        e['targets']=np.repeat('R4',len(e['state']));e['times']=np.arange(len(e['state']),dtype=np.float64)*.05
        indices['train'].extend((i,s) for s in e['starts'])
    heldout=copy.deepcopy(records[0]);heldout.update(episode_id='synthetic_heldout',split='validation');records.append(heldout)
    indices['validation']=[(2,0)]
    rgb=np.zeros((100,2,2,3),np.uint8)
    base=SimpleNamespace(episodes=records,indices=indices,dynamics={'synthetic':True},manifest={'synthetic':True},
        data_hash='synthetic_parent',normalization={'state':{'synthetic':'unchanged'},'encoding_binding':{'synthetic':'unchanged'},
            'residual_actions':OLD.fit_stats(records[:2]),'action_encoding_binding':OLD.action_encoding_binding()},parent=SimpleNamespace(rgb=lambda i:rgb))
    return base

def parent_api(monkeypatch,data):
    if hasattr(data,'EXPECTED_PARENT_SOURCES'):
        calls=[]
        def obs(rgb,state,phase,target,tokenizer,normalization):
            calls.append((rgb,state,phase,target,tokenizer,normalization))
            return {'state':state.copy(),'rgb':rgb.copy()}
        api=SimpleNamespace(core_sources=lambda:dict(data.EXPECTED_PARENT_SOURCES),observation=obs,
            stack_observations=lambda rows:{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        monkeypatch.setattr(data,'_parent_module',lambda:api)
        return calls

def test_views_keep_arrays_indices_and_state_bindings(monkeypatch,data):
    parent_api(monkeypatch,data);base=parent_fixture();a=data.view_parent(base,A);b=data.view_parent(base,B)
    for view in (a,b):
        assert view.parent is base and view.episodes is base.episodes and view.indices is base.indices
        assert view.manifest is base.manifest and view.dynamics is base.dynamics
        assert view.normalization['state'] is base.normalization['state']
        assert view.normalization['encoding_binding'] is base.normalization['encoding_binding']
        assert view.rgb(0) is base.parent.rgb(0)
        assert view.normalization['residual_actions']['fit_episode_ids']==['synthetic_0','synthetic_1']
    assert len({a.data_hash,b.data_hash,base.data_hash})==3
    assert data.dataset_binding(a)!=data.dataset_binding(b)
    data.validate_binding(a,data.dataset_binding(a))
    with pytest.raises(ValueError):data.validate_binding(a,data.dataset_binding(b))
    with pytest.raises(ValueError):data.validate_binding(a,{'data_hash':base.data_hash})

def test_batch_uses_original_observation_and_new_actions(monkeypatch,data):
    calls=parent_api(monkeypatch,data);base=parent_fixture();view=data.view_parent(base,B)
    obs,actions=data.make_batch(view,'train',[0,1],object())
    assert len(calls)==2 and calls[0][1] is not None
    np.testing.assert_array_equal(obs['state'][0],base.episodes[0]['state'][0])
    for i,(_,start) in enumerate(base.indices['train'][:2]):
        ep=base.episodes[0];restored=data.decode_prediction(view,actions[i],ep['state'][start],str(ep['phases'][start]))
        np.testing.assert_allclose(restored,ep['actions'][start:start+10],rtol=0,atol=2e-6)
    assert all(c[-1] is view.normalization for c in calls)

@pytest.mark.parametrize('mutation',[lambda p:p.indices['train'].append(p.indices['train'][0]),
    lambda p:p.indices['train'].append((2,0)),lambda p:p.indices['train'].append((0,19)),
    lambda p:p.indices['train'].append((0,True)),lambda p:p.normalization['residual_actions']['actions_by_phase']['FREE']['low'].__setitem__(0,-999)])
def test_invalid_parent_indices_or_changed_parent_norm_rejected(monkeypatch,data,mutation):
    parent_api(monkeypatch,data);base=parent_fixture();mutation(base)
    with pytest.raises(ValueError):data.view_parent(base,B)

def test_rejects_changed_source_and_invalid_batch_requests(monkeypatch,data):
    parent_api(monkeypatch,data);base=parent_fixture();view=data.view_parent(base,A)
    for split,indices in (('bad',[0]),('train',[True]),('train',[-1]),('train',[]),('train',[10000])):
        with pytest.raises(ValueError):data.make_batch(view,split,indices,None)
    monkeypatch.setattr(data,'_parent_module',lambda:SimpleNamespace(core_sources=lambda:{}))
    with pytest.raises(ValueError):data.view_parent(base,A)

def test_loader_rejects_missing_wrong_manifest_before_parent(monkeypatch,data,tmp_path):
    # View feature call keeps the initial absent-module red result meaningful.
    parent_api(monkeypatch,data);data.view_parent(parent_fixture(),A)
    p=tmp_path/'manifest.json';p.write_text('{}')
    with pytest.raises(ValueError):data.load_dataset(p,A)
