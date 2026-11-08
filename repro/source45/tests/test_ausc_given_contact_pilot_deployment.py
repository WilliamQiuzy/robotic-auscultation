"""Pilot-only binding and real action-prefix mechanics; no physics claims."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

from test_ausc_given_contact_pose import reference, metadata


def module():
    return importlib.import_module('sim.ausc_policy.m2_given_contact_pilot_deployment')


def complete_packet():
    from sim.ausc_policy import m2_given_contact_pilot_train as pilot
    from sim.ausc_policy import m2_given_contact_runtime as rt
    evaluations=[dict(step=s,validation={'act':{'full_chunk_mae_rad':1./s}}) for s in range(500,5001,500)]
    return dict(schema=pilot.PILOT_SCHEMA,complete=True,completed_steps=5000,selected_step=5000,
        best=dict(path='/weights/best_pilot.pt',bytes=123,sha256='a'*64),
        task_contract=rt.task_contract(),model=rt.shared.native_spec(),evaluations=evaluations,
        scope={'train':{'subject_ids':['p000449'],'episode_ids':['source-train']},
               'validation':{'subject_ids':['p001005'],'episode_ids':['source-val']}},**pilot.FLAGS)


def test_checkpoint_binding_rejects_wrong_selected_step_and_formal_schema():
    m=module();value=complete_packet()
    assert m.validate_complete(value) is value
    with pytest.raises(ValueError,match='selected'):
        m.validate_complete(dict(value,selected_step=500))
    with pytest.raises(ValueError):
        m.validate_complete(dict(value,schema='m2_given_contact_formal_act_training_v1'))
    with pytest.raises(ValueError):
        m.validate_complete(dict(value,test_payloads_opened=True))


def test_predicted_chunk_is_bound_to_request_and_consumed_as_e10(tmp_path):
    m=module();ref=reference(tmp_path);meta=dict(metadata(ref),target_id='L1')
    q=np.arange(7,dtype=float)/100
    state=m.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    dep=dict(metadata=meta,episode_id='diagnostic',source_episode_id='source-train',
        model={'checkpoint':{'sha256':'a'*64}},source_config={'sha256':'b'*64})
    request=m.make_request(dep,1,q,ref.pose_base,1.,metadata=meta,
        history=np.stack([state,state]),history_times=[0.,.05],
        context=dict(observed_at_s=.05,physics_row=12,context=['diagnostic','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
    assert m.validate_request(request,dep,sequence=1) is request
    relative=np.arange(140,dtype=float).reshape(20,7)/10000
    response=m.response_for(request,relative)
    block=m.decode_response(response,request)
    from sim.ausc_policy.m2_act_controller import FixedOriginPrefix
    prefix=FixedOriginPrefix(block,observed_q=q,context=('diagnostic','L1',0,'FREE'),stamp=.05)
    for i in range(10):
        np.testing.assert_array_equal(prefix.pop(.05+i*.05,prefix.context),q+relative[i])
    assert prefix.remaining==0
    with pytest.raises(ValueError):prefix.pop(.55,prefix.context)
    bad=deepcopy(response);bad['binding']['checkpoint_sha256']='c'*64
    with pytest.raises(ValueError):m.decode_response(bad,request)
    bad=deepcopy(response);bad['origin'][0]+=.001
    with pytest.raises(ValueError):m.decode_response(bad,request)


def test_derivation_preserves_physics_and_source_but_removes_expert_capture(tmp_path):
    m=module();meta=metadata(reference(tmp_path));meta.update(subject_id='p000449',split='train',episode_id='source-train',target_id='L1')
    original=dict(experiment='geometry_qualification',base='/workspace/wam',physics={'position_iterations':64},
        isaac_python='/opt/wam-runtime/envs/isaacsim60/bin/python',
        fresh_expert={'plans':[{'file':'/original-plan','sha256':'d'*64}],
            'force_feedback':{'frozen':True},'raw_multimodal_capture':{'enabled':True}},
        m2_given_contact_control={'metadata':meta,'episode_id':'source-train'},
        compact_scene_storage={'batch_id':'old','sources':{'scene':{'sha256':'e'*64}}})
    frozen=deepcopy(original)
    model={'checkpoint':{'sha256':'a'*64},'complete':{'sha256':'f'*64}}
    derived=m.derive_config(original,source_config={'path':'/old.json','sha256':'b'*64},
        model=model,run_id='pilot-diagnostic',source_inventory={'root':'/new-runtime'},scope=complete_packet()['scope'])
    assert original==frozen and derived['physics']==original['physics']
    assert derived['fresh_expert']['force_feedback']==original['fresh_expert']['force_feedback']
    assert derived['fresh_expert']['plans']==original['fresh_expert']['plans']
    assert 'raw_multimodal_capture' not in derived['fresh_expert']
    assert 'm2_given_contact_control' not in derived
    route=derived[m.ROUTE_KEY]
    assert route['source_episode_id']=='source-train' and route['metadata']['episode_id']!='source-train'
    assert route['schema']==m.DEPLOYMENT_SCHEMA and route['diagnostic_only'] is True
    assert derived['compact_scene_storage']['sources']==original['compact_scene_storage']['sources']


def test_router_selects_pilot_without_invoking_formal_or_teacher(monkeypatch):
    m=module()
    from sim.ausc_policy import fresh_expert
    marker=object()
    monkeypatch.setattr(m,'run_native_probe',lambda *args:marker)
    assert fresh_expert.run_probe(None,None,None,{m.ROUTE_KEY:{}},None) is marker


def test_preflight_authenticates_actual_source_and_rejects_config_or_test_changes(tmp_path):
    m=module();ref=reference(tmp_path)
    meta=dict(metadata(ref),subject_id='p000449',split='train',episode_id='source-train',
        target_id='L1',handoff_policy_id=m.runtime.handoff.POLICY_ID)
    original=dict(isaac_python=m.NATIVE_PYTHON,physics={'position_iterations':64},
        fresh_expert={'plans':[ref.snapshot()['sources']['fresh_plan']], 'raw_multimodal_capture':{'enabled':True}},
        m2_given_contact_control={'metadata':meta,'episode_id':'source-train'})
    source=m.runtime.write_new(tmp_path/'source.json',original)
    weights=tmp_path/'weights.pt';weights.write_bytes(b'no-deserialization-during-preflight')
    complete=complete_packet();complete['best']=m.runtime.reference(weights)
    complete_ref=m.runtime.write_new(tmp_path/'COMPLETE.json',complete)
    derived=m.derive_config(original,source_config=source,model={'checkpoint':complete['best'],'complete':complete_ref},
        run_id='pilot-diagnostic',source_inventory=m.sources(),scope=complete['scope'])
    assert m.preflight(derived)['reference'].sha256==ref.sha256
    from sim.ausc_policy import scene_store
    assert m.ROUTE_KEY not in scene_store._authoring(derived)
    bad=deepcopy(derived);bad['physics']['position_iterations']=32
    with pytest.raises(ValueError,match='binding'):m.preflight(bad)
    bad=deepcopy(derived);bad[m.ROUTE_KEY]['metadata']['split']='test'
    with pytest.raises(ValueError,match='TRAIN/VAL'):m.preflight(bad)
