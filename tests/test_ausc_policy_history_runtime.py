import copy
from types import SimpleNamespace
import numpy as np
import pytest
from sim.ausc_policy import act_baseline as a, act_runtime as r, history_runtime as h
from sim.ausc_policy.history_fit import make_packer,history_stats,history_deltas


def fixture():
    state=np.arange(19,dtype=np.float32)/20;state[17:]=[-1,-1]
    frames=[dict(index=i,time_s=i*.05,target='R1',q=(state[:7]+(i-12)*.001).tolist()) for i in range(13)]
    receipt=dict(schema='ausc_act_runtime_receipt_v1',scope='inference_only',device='cpu',upstream_commit=a.UPSTREAM_COMMIT,
        action_representation=a.BLOCK_RELATIVE,returned_representation=a.ABSOLUTE,state_dim=19,horizon=10,execute_steps=5,period_s=.05,
        history_variant='causal_history',environment_width=40,auxiliary_width=28,history_offsets_frames=list(h.OFFSETS))
    for key in ('checkpoint_sha256','normalization_sha256','source_config_sha256','policy_config_sha256','action_contract_sha256',
                'history_contract_sha256','history_preparation_sha256','paired_owner_sha256','paired_complete_sha256'):receipt[key]='a'*64
    request=r.build_request(state,receipt,request_id='test',task_id='one',episode_id='episode',frame_id=12,target_revision=0,
        sim_stamp_s=.6,target_id='R1',phase='FREE')
    request.update(schema='ausc_act_history_request_v1',history=h.history_from_frames(frames,frames[-1]))
    return state,frames,receipt,request


@pytest.mark.parametrize('variant',['repeat_current','causal_history'])
def test_live_input_is_exactly_training_packer_and_anchor_once(variant):
    state,frames,receipt,request=fixture();q=h.validate_history(request['history'],request)
    history=np.zeros((5,19),np.float32);history[:,:7]=q;history[-1]=state
    aux=history_stats(history_deltas(history[None],'causal_history'))
    stats={key:dict(low=[-1.]*size,high=[1.]*size) for key,size in [('state',19),('action',7)]}
    rows=[dict(state=state,episode_id='e',start=2)]
    expected=make_packer({('e',2):history},aux,variant)(rows,stats,training=False)
    actual=h.pack_history(state,q,stats,aux,variant)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    receipt['history_variant']=variant;request['receipt_sha256']=r.digest_json(receipt)
    engine=SimpleNamespace(predict=lambda batch:np.zeros((1,10,7),np.float32))
    artifacts=SimpleNamespace(receipt=receipt,stats=stats)
    response=h.InferenceService(artifacts,aux,engine).handle(request)
    h.validate_response(response,request,receipt)
    np.testing.assert_array_equal(response['raw'],np.broadcast_to(state[:7],(10,7)))
    changed=copy.deepcopy(request);changed['history'][0]['q'][0]+=.001
    with pytest.raises(ValueError):h.validate_response(response,changed,receipt)


@pytest.mark.parametrize('change',['future','wrong_target','wrong_index','missing','q_mismatch','nonfinite','extra','wrong_schema'])
def test_history_request_rejects_causality_identity_and_anchor_faults(change):
    _,_,receipt,request=fixture()
    if change=='future':request['history'][0]['sim_stamp_s']=.7
    if change=='wrong_target':request['history'][0]['target_id']='L1'
    if change=='wrong_index':request['history'][0]['frame_id']=3
    if change=='missing':request['history'].pop()
    if change=='q_mismatch':request['history'][-1]['q'][0]=.3
    if change=='nonfinite':request['history'][0]['q'][0]=float('nan')
    if change=='extra':request['history'][0]['action']=[0]*7
    if change=='wrong_schema':request['schema']='ausc_act_request_v1'
    with pytest.raises(ValueError):h.validate_request(request,receipt)


def test_observed_history_cannot_be_padded_or_duplicate():
    _,frames,_,request=fixture()
    values=h.history_from_frames(frames,frames[-1]);assert [v['frame_id'] for v in values]==[2,7,10,11,12]
    with pytest.raises(ValueError):h.history_from_frames(frames[-5:],frames[-1])
    altered=copy.deepcopy(frames);altered[-2]['index']=12
    with pytest.raises(ValueError):h.history_from_frames(altered,altered[-1])


def test_legacy_receipt_cannot_enable_history():
    _,_,receipt,_=fixture();receipt.pop('history_variant')
    with pytest.raises(ValueError):h.validate_receipt(receipt)
