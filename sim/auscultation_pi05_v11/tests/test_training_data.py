import copy
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v11 import training_data as D
from sim.auscultation_pi05_v7 import training_data as O


def fixture(monkeypatch):
    ep=dict(episode_id='R5_x',target='R5',split='validation',state=np.zeros((10,41)),actions=np.zeros((10,7)),phases=np.array(['FREE']*10),targets=np.array(['R5']*10))
    d=SimpleNamespace(episodes=[ep],loaded_scopes=('validation',),indices={'validation':[(0,0)]},normalization={'state':'unchanged'},rgb=lambda i:np.zeros((10,2,2,3),np.uint8))
    monkeypatch.setattr(D,'_state_ok',lambda _:None)
    monkeypatch.setattr(O,'observation',lambda rgb,raw,phase,target,tok,norm:dict(rgb=rgb,state=raw,target=target))
    monkeypatch.setattr(O,'stack_observations',lambda obs:obs)
    return d


def test_heldout_inputs_never_touch_actions_or_codec(monkeypatch):
    d=fixture(monkeypatch);del d.episodes[0]['actions']
    def forbidden(*args):raise AssertionError('Heldout inverse or encode called')
    monkeypatch.setattr(D.codec,'inverse_coordinates',forbidden);monkeypatch.setattr(D.codec,'encode_chunk',forbidden)
    obs=D.make_observation_batch(d,'validation',[0],None)
    assert len(obs)==1 and obs[0]['target']=='R5'
    with pytest.raises(ValueError,match='TRAIN'):D.make_batch(d,'validation',[0],None)


@pytest.mark.parametrize('bad',['row','chunk','phase'])
def test_row_target_or_chunk_mismatch_rejected(monkeypatch,bad):
    d=fixture(monkeypatch)
    if bad=='row':d.episodes[0]['target']='R4'
    elif bad=='chunk':d.episodes[0]['targets'][9]='L5'
    else:d.episodes[0]['phases'][9]='HOLD'
    with pytest.raises(ValueError):D.make_observation_batch(d,'validation',[0],None)


def test_test_auth_required_before_design_or_arrays(monkeypatch):
    def forbidden(*args):raise AssertionError('Opened data before test authorization')
    monkeypatch.setattr(D,'design',forbidden)
    with pytest.raises(ValueError,match='authorized'):D.load_dataset('absent',scopes=['test'],normalization_path='absent',expected_normalization_sha256='0'*64)


def test_original_membership_does_not_follow_new_sixteen_fit_ids():
    d=SimpleNamespace(_identity={'parent_dataset_binding':{'normalization_fit_episode_ids':['original']}},normalization={'residual_actions':{'fit_episode_ids':['original','added']}})
    assert D.original_ids(d)==['original']
