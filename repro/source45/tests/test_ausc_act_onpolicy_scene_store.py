"""Invented fixed8 metadata and scene bytes; no data/model/Isaac execution."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_act_onpolicy_collection as collection, scene_store


def collection_config(tmp_path, monkeypatch):
    from test_ausc_policy_m2_pilot_config import fixture
    from sim.ausc_policy import m2_pilot_config, m2_stop_residual_trajectory as residual
    request=collection.planned_requests()[0]
    refs,plan,save=fixture(tmp_path,monkeypatch,identity=request['subject_id'])
    patient=json.loads(Path(refs['patient_receipt']['file']).read_text())
    patient['targets'][0]['name']=request['target_id']
    patient['proposal']=request['subject_id']+'_'+request['pose_id']
    refs['patient_receipt']=save('patient_receipt.json',patient)
    complete=json.loads(Path(refs['prepared_complete']['file']).read_text())
    complete['patient']=patient
    refs['prepared_complete']=save('COMPLETE.json',complete)
    prepared=json.loads(Path(refs['preparation_config']['file']).read_text())
    prepared.update(physics=deepcopy(collection.recovery.PHYSICS),isaac_python=collection.recovery.NATIVE_PYTHON,
        approved_camera=dict(index=7),table=dict(height_m=.7))
    refs['preparation_config']=save('prep.json',prepared)
    value=json.loads(Path(plan['file']).read_text())
    value.update(name=request['target_id'],command_rotation_world=np.diag([1.,-1.,-1.]).tolist(),
        command_normal_world=[0.,0.,1.])
    value['patient_provenance']['binding']=refs['patient_receipt']
    plan=save('fresh_plan.json',value)
    for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):
        (tmp_path/name).write_text('{}')
    original=m2_pilot_config.build_given_contact_expert(**refs,fresh_plan=plan,episode_id='original-fixed8',
        scene_store=dict(root=str(tmp_path/'cas'),batch_id='original-fixed8'))['config']
    original[residual.CONFIG_KEY]=deepcopy(residual.CONFIG)
    source=collection.rt.write_new(tmp_path/'nominal.json',original)
    # Unavailable archived preparation and warm-export proofs are leaf fixtures.
    # Keep the public preflight, deployment validator, exact projection, teacher
    # binding and scene-store authoring/payload checks real.
    monkeypatch.setattr(collection,'_prepared',lambda ref:{})
    initialization=dict(state_dict=dict(path='/invented/no-weights.pt',bytes=1,sha256='a'*64))
    monkeypatch.setattr(collection.warm,'validate_receipt',lambda ref:deepcopy(initialization))
    nominal=collection.nominal_references();nominal[0]=source
    monkeypatch.setattr(collection,'nominal_references',lambda:deepcopy(nominal))
    inventory=dict(root='/invented/source',code_commit='fixture',files={})
    monkeypatch.setattr(collection,'sources',lambda:deepcopy(inventory))
    episode='act11-onpolicy-'+request['slot_id']+'-fixture'
    dep=dict(schema=collection.DEPLOYMENT_SCHEMA,prepared=deepcopy(collection.PREPARED),request=request,
        source_config=source,initialization=dict(path='/invented/warm.json',bytes=1,sha256='b'*64),
        episode_id=episode,source_episode_id='original-fixed8',metadata={},source_inventory=deepcopy(inventory),
        runtime_config_sha256='',model=dict(schema=collection.warm.COLLECTION_SCHEMA,
            checkpoint=initialization['state_dict'],benchmark_model=False),**collection.FLAGS)
    config=collection._projection(original,dep)
    dep['metadata']=deepcopy(config['m2_given_contact_control']['metadata'])
    dep['runtime_config_sha256']=collection.rt.digest(config)
    config[collection.ROUTE_KEY]=dep
    collection.preflight(config)
    return config,prepared


def test_collection_authoring_projects_to_exact_original_prepared_scene(tmp_path,monkeypatch):
    config,prepared=collection_config(tmp_path,monkeypatch);before=deepcopy(config)
    projected=scene_store._authoring(config)
    assert collection.ROUTE_KEY not in projected
    assert projected==scene_store._authoring(prepared)
    assert config==before


def test_collection_reuses_exact_prepared_binding_with_attached_capture(tmp_path,monkeypatch):
    config,_=collection_config(tmp_path,monkeypatch)
    binding,complete=scene_store.validate_config(config)
    assert binding==config['compact_scene_storage'] and complete['completed'] is True
    capture=config['fresh_expert']['raw_multimodal_capture']
    assert capture['enabled'] is True and capture['profile']=='compact'
    assert capture['compact']['metadata']==config[collection.ROUTE_KEY]['metadata']


def test_collection_authentication_precedes_every_scene_store_payload_read(tmp_path,monkeypatch):
    config,_=collection_config(tmp_path,monkeypatch);calls=[];authenticated=False
    preflight=collection.preflight;read=scene_store.read_source
    def authenticate(value,**kwargs):
        nonlocal authenticated
        calls.append('preflight')
        result=preflight(value,**kwargs);authenticated=True
        return result
    def observe(ref,**kwargs):
        assert authenticated, 'Collection authentication must precede patient/USD/NPZ binding reads'
        calls.append(Path(ref['file']).name)
        return read(ref,**kwargs)
    monkeypatch.setattr(collection,'preflight',authenticate)
    monkeypatch.setattr(scene_store,'read_source',observe)
    binding,complete=scene_store.validate_config(config)
    assert calls[0]=='preflight' and {'patient_receipt.json','candidate_physics.usdc','collision_geometry.npz'}<=set(calls)
    assert binding==config['compact_scene_storage'] and complete['completed'] is True


@pytest.mark.parametrize('fault',['schema','prepared','stale_source','forged_projection','detached_capture',
    'full_capture','ordinary_act','paired','recovery'])
def test_bad_collection_route_rejects_before_scene_store_payload(tmp_path,monkeypatch,fault):
    config,_=collection_config(tmp_path,monkeypatch);dep=config[collection.ROUTE_KEY]
    if fault=='schema':dep['schema']='unknown'
    elif fault=='prepared':dep['prepared']['sha256']='0'*64
    elif fault=='stale_source':dep['source_inventory']['code_commit']='stale'
    elif fault=='forged_projection':dep['runtime_config_sha256']='0'*64
    elif fault=='detached_capture':config['fresh_expert'].pop('raw_multimodal_capture')
    elif fault=='full_capture':config['fresh_expert']['raw_multimodal_capture']['profile']='full_raw'
    elif fault=='ordinary_act':config['m2_act_deployment']={}
    elif fault=='paired':config['matched_scene_eval']={}
    else:config[collection.recovery.ROUTE_KEY]={}
    seen=[]
    def forbidden(ref,**kwargs):
        seen.append(ref)
        pytest.fail('Invalid collection route reached scene-store payload')
    monkeypatch.setattr(scene_store,'read_source',forbidden)
    with pytest.raises(ValueError):scene_store.validate_config(config)
    assert not seen


@pytest.mark.parametrize('field',['geometry_source_sha256','approved_camera','table','physics'])
def test_collection_authentication_does_not_hide_authoring_drift(tmp_path,monkeypatch,field):
    config,_=collection_config(tmp_path,monkeypatch)
    config[field]='changed'
    with pytest.raises(ValueError,match='projection'):scene_store._authoring(config)


@pytest.mark.parametrize('field',['geometry_source_sha256','approved_camera','table','physics'])
def test_original_prepared_authoring_comparison_remains_exact(tmp_path,monkeypatch,field):
    config,_=collection_config(tmp_path,monkeypatch)
    source=config['compact_scene_storage']['sources']['preparation_config']
    prepared=json.loads(scene_store.read_source(source));prepared[field]='changed'
    Path(source['file']).write_text(json.dumps(prepared))
    source.update(scene_store.source_ref(source['file']))
    # The nominal runtime itself still authenticates, but its prepared authoring
    # must independently match at the final original scene-store comparison.
    nominal=config[collection.ROUTE_KEY]['source_config']
    original=json.loads(Path(nominal['path']).read_text())
    original['compact_scene_storage']['sources']['preparation_config']=deepcopy(source)
    Path(nominal['path']).write_text(json.dumps(original))
    nominal.update(collection.rt.reference(nominal['path']))
    config[collection.ROUTE_KEY]['runtime_config_sha256']=collection.rt.digest({
        k:v for k,v in config.items() if k!=collection.ROUTE_KEY})
    with pytest.raises(ValueError,match='Runtime authoring differs'):scene_store.validate_config(config)


@pytest.mark.parametrize('route',['m2_act_deployment','m2_given_contact_deployment',
    'm2_given_contact_pilot_deployment','matched_scene_eval'])
def test_collection_registration_does_not_relax_ordinary_capture_routes(tmp_path,monkeypatch,route):
    from test_ausc_policy_scene_store import bound
    _,config=bound(tmp_path)
    config[route]={}
    if route!='m2_act_deployment':
        from sim.ausc_policy import m2_given_contact_deployment, m2_given_contact_pilot_deployment, m2_rebuild_paired_eval
        module={'m2_given_contact_deployment':m2_given_contact_deployment,
            'm2_given_contact_pilot_deployment':m2_given_contact_pilot_deployment,
            'matched_scene_eval':m2_rebuild_paired_eval}[route]
        monkeypatch.setattr(module,'preflight_arm' if route=='matched_scene_eval' else 'preflight',lambda *a,**k:{})
    with pytest.raises(ValueError,match='capture'):scene_store.validate_config(config)
