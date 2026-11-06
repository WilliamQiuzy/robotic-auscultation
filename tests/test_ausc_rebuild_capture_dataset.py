"""CPU contract tests; synthetic arrays here cannot qualify a native episode."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt


def helper():
    name='sim.ausc_policy.m2_rebuild_capture_dataset'
    assert importlib.util.find_spec(name) is not None, 'Fresh rebuild capture/export helper missing'
    return importlib.import_module(name)


def request():
    ref=dict(path='/not-opened.json',bytes=2,sha256='a'*64)
    subjects=['p001120','p000758','p000923']
    poses=['p000474_a001230_f0000','p000476_a001231_f0023']
    scenes=[]
    for subject in subjects:
        for pose in poses:
            scenes.append(dict(subject_id=subject,split='validation' if subject=='p000923' else 'train',
                cohort='selection_validation' if subject=='p000923' else 'train',pose_id=pose,
                preparation_config=ref,prepared_complete=ref,patient_receipt=ref,
                plans={'R3':dict(status='unavailable',reason='planner not run'),
                       'L3':dict(status='unavailable',reason='planner not run')}))
    return dict(schema='m2_rebuild_expert_capture_request_v1',run_id='rebuild-test',base='/workspace/wam',
        source=dict(code_commit='1'*40,code_commit_file=ref,source_manifest=ref),
        reconstruction_source=ref,subject_ids=subjects,targets=['R3','L3'],scenes=scenes,
        parallelism=2,time_limit_seconds=3600,max_frames=5000,
        retainer_witness_contract={'qualification':{'file':'/not-opened.json','sha256':'a'*64}},
        scene_store=dict(root='/workspace/wam/scenes/rebuild-test',batch_id='rebuild-test'))


def samples(count=2,offset=0.):
    state=np.zeros((count,17));state[:,0]=np.arange(count)*.01+offset
    return dict(states=state,histories=np.stack([state,state],axis=1),
        actions=np.full((count,20,7),offset),origins=state[:,:7].copy(),velocities=np.zeros((count,7)),
        observation_times=np.arange(count)*.05,schema=rt.given.DATASET_SCHEMA,
        task_id=rt.pose.TASK_ID,action_semantics_id=rt.given.ACTION_SEMANTICS_ID,
        observation_contract=rt.pose.observation_contract(),derived_index=dict(windows=[
            dict(observation_indices=[i,i+1],action_record_indices=[i+1,i+20]) for i in range(count)]))


def decoded(episode,split,subject,count,offset):
    row=dict(episode_id=episode,split=split,subject_id=subject,target_id='R3',target='R3',config={},
        pose_id='p000474_a001230_f0000',attempt_id=episode)
    receipt=dict(row,windows=count)
    return row,(samples(count,offset),receipt,dict(reference_sha256='a'*64,given_reference={}))


def test_request_has_explicit_variable_denominator_and_no_frozen_payload_access():
    m=helper();value=request();rows=m.validate_request(value)
    assert len(rows)==12
    assert [(r['subject_id'],r['pose_id'],r['target_id']) for r in rows]==[
        (s['subject_id'],s['pose_id'],t) for s in value['scenes'] for t in value['targets']]
    assert len({r['episode_id'] for r in rows})==12


@pytest.mark.parametrize('alter',[
    lambda v:v['subject_ids'].append('p001030'),
    lambda v:v['scenes'][0].update(subject_id='p000956'),
    lambda v:v['scenes'][0].update(cohort='frozen_evaluation'),
    lambda v:v['scenes'][0].update(split='validation'),
    lambda v:v['scenes'].pop(),
    lambda v:v['scenes'].append(deepcopy(v['scenes'][0])),
    lambda v:v['scenes'][0]['plans'].pop('L3'),
    lambda v:v.update(targets=['R3','R3']),
    lambda v:v.update(parallelism=8),
    lambda v:v.update(physics_gate_override=True),
])
def test_rejects_scope_changes_before_opening_any_scene_or_payload(alter):
    m=helper();value=request();alter(value)
    with pytest.raises(ValueError):m.validate_request(value)


def test_unavailable_plans_stay_in_batch_denominator_without_fake_jobs(monkeypatch,tmp_path):
    m=helper();value=request()
    monkeypatch.setattr(m,'validate_sources',lambda value:None)
    monkeypatch.setattr(m,'scene_metadata',lambda scene,value:None)
    result=m.build_batch(value,tmp_path/'build')
    assert result['fixed_denominator']==12 and len(result['attempts'])==12
    assert result['runnable_cases']==0 and result['batch'] is None
    assert all(row['preparation_status']=='unavailable' for row in result['attempts'])
    assert result['physical_validation'] is False


def test_comparison_scope_is_frozen_before_any_model_or_evaluation(monkeypatch,tmp_path):
    m=helper();value=request()
    monkeypatch.setattr(m,'validate_sources',lambda value:None)
    monkeypatch.setattr(m,'scene_metadata',lambda scene,value:None)
    result=m.build_batch(value,tmp_path/'build')
    scope=rt.read_bound(result['comparison_scope'])
    assert scope['arms']==['expert','act','astra'] and scope['frozen_before_evaluation'] is True
    assert scope['fixed_denominator']==12 and len(scope['cases'])==12
    assert scope['request']==result['request']
    assert scope['frozen_evaluation_subject_ids_excluded']==['p001030','p000956']
    assert all(row['config'] is None for row in scope['cases'])


def test_batch_uses_original_builder_without_mutating_its_physics(monkeypatch,tmp_path):
    m=helper();value=request();scene=value['scenes'][0]
    header=rt.write_new(tmp_path/'header.json',{})
    for key in ('preparation_config','prepared_complete','patient_receipt'):scene[key]=header
    plan=rt.write_new(tmp_path/'plan.json',dict(success=True,name='R3'))
    scene['plans']['R3']=dict(plan=plan)
    monkeypatch.setattr(m,'validate_sources',lambda value:None)
    monkeypatch.setattr(m,'scene_metadata',lambda scene,value:None)
    expected=dict(experiment='geometry_qualification',base='/workspace/wam',time_limit_seconds=1200,
        isaac_python='/isaac/bin/python',ld_library_path='/isaac/lib',
        fresh_expert=dict(force_gate_n=8.,parking='unchanged'),m2_given_contact_control=dict(schema=rt.pose.TASK_ID))
    calls=[]
    def assemble(**kwargs):
        calls.append(kwargs);return dict(config=deepcopy(expected),receipt=dict(physical_validation=False))
    monkeypatch.setattr(m.pilot,'build_given_contact_expert',assemble)
    result=m.build_batch(value,tmp_path/'build')
    batch=rt.read_bound(result['batch']);assert len(batch['jobs'])==1
    assert rt.read_bound(result['attempts'][0]['config'])==expected
    assert calls[0]['retainer_witness_contract']==value['retainer_witness_contract']
    assert calls[0]['scene_store']==value['scene_store']
    assert len(result['attempts'])==12


def test_failed_scene_preparation_is_retained_without_opening_missing_refs(monkeypatch,tmp_path):
    m=helper();value=request();scene=value['scenes'][0]
    scene.update(preparation_error='Original scene preparation failed: collision mesh export',
        preparation_config=None,prepared_complete=None,patient_receipt=None)
    monkeypatch.setattr(m,'validate_sources',lambda value:None)
    seen=[]
    def check(scene,value):
        assert scene.get('preparation_error') is None
        seen.append(scene['subject_id'])
    monkeypatch.setattr(m,'scene_metadata',check)
    result=m.build_batch(value,tmp_path/'build')
    assert len(seen)==5 and result['fixed_denominator']==12
    assert all(row['preparation_status']=='scene_preparation_failed' and 'mesh export' in row['reason']
        for row in result['attempts'][:2])


def test_failed_headers_and_original_physical_rejection_stay_in_ledger(monkeypatch):
    m=helper();rows=[decoded('good','train','p001120',2,.01)[0],
        decoded('unsafe','train','p000758',2,.02)[0],decoded('absent','validation','p000923',2,.03)[0]]
    calls=[]
    def read(row,base,context):
        calls.append(row['episode_id'])
        if row['episode_id']=='unsafe':raise ValueError('original contact/parking physical gate failed')
        return decoded('good','train','p001120',2,.01)[1]
    original=m.data._physical
    monkeypatch.setattr(m.data,'_read_payload',read)
    entries=[dict(case=r,row=r,context=None,header_error=None) for r in rows]
    entries[2]['header_error']='missing child COMPLETE'
    decoded_rows,outcomes=m.postflight(entries,Path('/workspace/wam'))
    assert calls==['good','unsafe'] and len(decoded_rows)==1 and len(outcomes)==3
    assert [r['positive_data'] for r in outcomes]==[True,False,False]
    assert 'physical gate' in outcomes[1]['error']
    assert m.data._physical is original


def test_fresh_combination_and_official_loader_need_no_original_eight_arrays(tmp_path):
    m=helper();rows=[decoded('a','train','p001120',3,.01),decoded('b','validation','p000923',2,.04)]
    combined=m.combine(rows)
    assert combined['counts']=={'train':3,'validation':2}
    assert combined['episodes'][0]['observed_pose_id']=='p000474_a001230_f0000'
    assert combined['episodes'][0]['source_attempt_id']=='a'
    assert np.array_equal(combined['train']['actions'],rows[0][1][0]['actions'])
    assert combined['normalization']==rt.given.fit_normalization(combined['train'],split='train')
    ref=m.publish_arrays(tmp_path/'dataset',combined,dict(fixed_denominator=4),[],{})
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as trainer
    prepared=trainer.load_dataset(ref)
    assert prepared['train']['histories'].shape==(3,2,17)
    assert prepared['validation']['actions'].shape==(2,20,7)
    config=m.training_config('/workspace/wam',ref,steps=1000,eval_every=500,seed=20261105)
    assert trainer.validate_config(config)==config
    assert config['q_noise_std']==config['q_dropout']==0.
    assert set(rt.read_bound(ref)['arrays'])=={'train','validation'}


@pytest.mark.parametrize('rows',[
    [decoded('a','train','p001120',2,.01)],
    [decoded('a','train','p001120',2,.01),decoded('b','validation','p001030',2,.02)],
    [decoded('a','train','p001120',2,.01),decoded('a','validation','p000923',2,.02)],
])
def test_combination_refuses_missing_selection_frozen_ids_or_duplicate_episode(rows):
    with pytest.raises(ValueError):helper().combine(rows)
