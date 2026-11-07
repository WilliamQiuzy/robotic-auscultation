"""CPU configuration and clock tests; no policy or native rollout evidence."""
from copy import deepcopy
import hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
from test_ausc_policy_m2_contract import metadata
from sim.ausc_policy import m2_contract as contract


def module():
    from sim.ausc_policy import m2_estimate_provider
    return m2_estimate_provider


def sources(tmp_path):
    from sim.ausc_policy.m2_pilot_config import target_estimate
    target=dict(position_world_m=[0.,0.,0.],contact_plane_normal_world=[0.,0.,1.])
    pose,noise=target_estimate(target,np.eye(4),4)
    meta=dict(metadata(),target_source='gt_plus_measured_noise',subject_id='p000439',split='train',
        identity_split_sha256=contract.IDENTITY_SPLIT_SHA256,
        target_id='L1',patient_asset_sha256='a'*64,installed_patient_receipt_sha256='b'*64,
        source_kind='isaac_live_physics',replay=False)
    compact=dict(metadata=meta,estimated_target_base=pose.tolist(),target_confidence=.9,target_estimate_source_id='seeded4')
    doc=dict(fresh_expert=dict(raw_multimodal_capture=dict(profile='compact',compact=compact)))
    receipt=dict(subject_id='p000439',split='train',patient_receipt=dict(file='/bound/patient.json',sha256='b'*64),target_estimate_noise=noise)
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config=dict(schema=module().SCHEMA,compact_config=write('compact.json',doc),
        noise_receipt=write('noise.json',receipt),revision=0)
    patient=dict(identity='p000439',split='train',asset_sha256='a'*64,receipt_sha256='b'*64)
    return config,meta,patient,write,doc,receipt


def provider(tmp_path):
    config,meta,patient,*_=sources(tmp_path)
    return module().FrozenEstimateProvider(config,metadata=meta,episode_id='new-episode',target_id='L1',patient=patient)


def controller(state):
    return SimpleNamespace(episode_id='new-episode',target=SimpleNamespace(name='L1'),env=SimpleNamespace(state=lambda:deepcopy(state)))


def test_provider_returns_same_time_q_and_frozen_estimate_not_teacher_target(tmp_path):
    p=provider(tmp_path);state=dict(q=np.zeros(7),time_s=.05)
    c=controller(state)
    row=p(c,state)
    assert row['episode_id']=='new-episode' and row['observed_at_s']==.05 and row['revision']==0
    np.testing.assert_array_equal(row['measured_q'],state['q'])
    assert row['source_id']=='seeded4' and row['phase']=='FREE'
    assert p.snapshot()['physics_executed'] is False


def test_phase_latches_on_estimated_tcp_distance_without_phase_feature(tmp_path,monkeypatch):
    p=provider(tmp_path);state=dict(q=np.zeros(7),time_s=0.);c=controller(state)
    distance=[.10]
    real=contract.encode_state
    def encode(*args,**kwargs):
        result=real(*args,**kwargs);result[7:10]=[distance[0],0,0];return result
    monkeypatch.setattr(contract,'encode_state',encode)
    assert p(c,state)['phase']=='FREE'
    state['time_s']+=1/240;distance[0]=.089
    assert p(c,state)['phase']=='APPROACH'
    state['time_s']+=1/240;distance[0]=.20
    assert p(c,state)['phase']=='APPROACH'
    assert len(p.snapshot()['phase_transitions'])==1


@pytest.mark.parametrize('fault',['clock','q','target','episode','backwards'])
def test_provider_rejects_identity_or_same_time_feedback_mismatch(tmp_path,fault):
    p=provider(tmp_path);state=dict(q=np.zeros(7),time_s=.05);c=controller(state)
    p(c,state);observed=deepcopy(state)
    if fault=='clock':observed['time_s']+=1/240
    if fault=='q':observed['q'][0]=.1
    if fault=='target':c.target.name='R1'
    if fault=='episode':c.episode_id='other'
    if fault=='backwards':state['time_s']=observed['time_s']=.04
    with pytest.raises(ValueError):p(c,observed)


@pytest.mark.parametrize('fault',['hash','noise','normal','patient','missing'])
def test_provider_refuses_unbound_or_invented_estimate(tmp_path,fault):
    config,meta,patient,write,doc,receipt=sources(tmp_path)
    if fault=='hash':config['noise_receipt']['sha256']='f'*64
    if fault=='noise':receipt['target_estimate_noise']['position_bias_base_m'][0]+=.001
    if fault=='normal':receipt['target_estimate_noise']['normal_noise_applied']=True
    if fault=='patient':patient['receipt_sha256']='c'*64
    if fault=='missing':doc['fresh_expert']['raw_multimodal_capture']['compact'].pop('estimated_target_base')
    if fault!='hash':
        config['noise_receipt']=write('noise.json',receipt);config['compact_config']=write('compact.json',doc)
    with pytest.raises(ValueError):module().FrozenEstimateProvider(config,metadata=meta,episode_id='new-episode',target_id='L1',patient=patient)


@pytest.mark.parametrize('key,value',[
    ('subject_id','p000452'),('split','test'),('target_id','R5'),
    ('identity_split_sha256','c'*64),('patient_asset_sha256','c'*64),
    ('installed_patient_receipt_sha256','c'*64),('episode_id','other-episode'),
    ('source_kind','static_render'),('replay',True),('replay',0)])
def test_deployment_cannot_contradict_authenticated_identity(tmp_path,key,value):
    config,meta,patient,*_=sources(tmp_path)
    supplied=dict(meta,**{key:value})
    with pytest.raises(ValueError,match='contradicts authenticated identity'):
        module().FrozenEstimateProvider(config,metadata=supplied,episode_id='new-episode',target_id='L1',patient=patient)


def test_minimal_deployment_metadata_materializes_identity_and_own_episode(tmp_path):
    config,meta,patient,write,doc,_=sources(tmp_path)
    doc['fresh_expert']['raw_multimodal_capture']['compact']['metadata']['episode_id']='source-episode'
    config['compact_config']=write('compact.json',doc)
    supplied=dict(metadata(),target_source='gt_plus_measured_noise')
    before=deepcopy(supplied)
    p=module().FrozenEstimateProvider(config,metadata=supplied,episode_id='new-episode',target_id='L1',patient=patient)
    assert supplied==before
    assert p.metadata['subject_id']==patient['identity'] and p.metadata['split']==patient['split']
    assert p.metadata['target_id']=='L1' and p.metadata['episode_id']=='new-episode'
    assert p.metadata['source_kind']=='isaac_live_physics' and p.metadata['replay'] is False
    assert p.snapshot()['source_metadata']['episode_id']=='source-episode'
    assert p.snapshot()['deployment_metadata']==p.metadata


@pytest.mark.parametrize('key,value',[('source_kind','static_render'),('replay',True),('replay',0)])
def test_frozen_source_cannot_claim_replay_or_nonphysics(tmp_path,key,value):
    config,meta,patient,write,doc,_=sources(tmp_path)
    doc['fresh_expert']['raw_multimodal_capture']['compact']['metadata'][key]=value
    config['compact_config']=write('compact.json',doc)
    with pytest.raises(ValueError,match='live non-replay'):
        module().FrozenEstimateProvider(config,metadata=meta,episode_id='new-episode',target_id='L1',patient=patient)
