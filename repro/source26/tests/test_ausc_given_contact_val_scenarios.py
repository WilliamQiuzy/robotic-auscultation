"""Metadata-only VAL-scenario adapter checks; no Isaac, model or TEST payloads."""
from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as expanded
from test_ausc_given_contact_pose import inputs, metadata


def module():
    name='sim.ausc_policy.m2_given_contact_expanded_val_scenarios'
    assert importlib.util.find_spec(name) is not None,'Separate validation scenario adapter missing'
    return importlib.import_module(name)


def source_config(root,subject,target,index):
    root.mkdir();point,plan_ref,patient_ref=inputs(root)
    patient=rt.read_bound(dict(path=patient_ref['file'],sha256=patient_ref['sha256']))
    patient.update(identity=subject,split='validation');patient['targets'][0]['name']=target
    patient_ref=rt.write_new(root/'val-patient.json',patient)
    bound_patient=dict(file=patient_ref['path'],sha256=patient_ref['sha256'])
    plan=rt.read_bound(dict(path=plan_ref['file'],sha256=plan_ref['sha256']))
    plan.update(name=target,patient_provenance=dict(identity=subject,split='validation',asset_sha256='a'*64,binding=bound_patient))
    plan_ref=rt.write_new(root/'val-plan.json',plan);bound_plan=dict(file=plan_ref['path'],sha256=plan_ref['sha256'])
    point.update(name=target,manifest_ref=bound_patient)
    ref=rt.pose.build_given_reference(point,bound_plan,bound_patient,np.eye(4),0.)
    episode='expert-'+str(index)
    meta=dict(metadata(ref),subject_id=subject,split='validation',target_id=target,episode_id=episode,
        handoff_policy_id=rt.handoff.POLICY_ID)
    config=dict(experiment='geometry_qualification',base='/workspace/wam',time_limit_seconds=3600,
        isaac_python=expanded.NATIVE_PYTHON,ld_library_path='/native/lib',
        physics=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02),
        fresh_expert=dict(plans=[bound_plan],force_feedback={'frozen':True},raw_multimodal_capture={'enabled':True}),
        m2_given_contact_control=dict(metadata=meta,episode_id=episode),
        **{k:meta[k] for k in ('geometry_id','geometry_source_sha256','freshly_generated')})
    return rt.write_new(root/'config.json',config),episode


def setup(tmp_path,monkeypatch):
    m=module();splits=json.loads(rt.shared.contract.IDENTITY_SPLIT_PATH.read_text())['splits']
    identities=['p000486','p000566','p000570','p000722','p000785','p000843','p001017','p001042','p001139','p100069']
    cases=[];targets=[s+str(i) for s in 'LR' for i in range(1,6)]
    for i in range(30):
        subject=identities[i//3];target=targets[i%10]
        config,episode=source_config(tmp_path/('case'+str(i)),subject,target,i)
        fresh=subject in ('p000486','p001139')
        cases.append(dict(index=i+1,scenario_id='val-'+str(i+1),episode_id=episode,subject_id=subject,
            split='validation',target=target,seed=0,config=config,independent_subject_holdout=fresh,
            validation_stratum='independent_subject_holdout' if fresh else 'checkpoint_selection_validation',
            tuning_permitted=not fresh,model_outcomes_consulted=False))
    source=rt.write_new(tmp_path/'EXPERT_CONFIGS.json',dict(schema=m.INPUT_SCHEMA,complete=True,fixed_denominator=30,
        cases=cases,diagnostic_only=True,formal_acceptance=False,test_payloads_opened=False,model_outcomes_consulted=False))
    weights=tmp_path/'best.pt';weights.write_bytes(b'actual-selected-weight-ref')
    complete_ref=rt.write_new(tmp_path/'COMPLETE.json',{'schema':'mock-completed-model-bound-by-expanded-adapter'})
    model={'checkpoint':rt.reference(weights),'complete':complete_ref}
    complete={'completed_steps':20000,'scope':{'train':{'subject_ids':splits['train'][:2]},
        'validation':{'subject_ids':[x for x in identities if x not in ('p000486','p001139')]}}}
    monkeypatch.setattr(expanded,'model_binding',lambda ref:(deepcopy(model),deepcopy(complete)))
    return m,source,model,complete,cases


def test_freeze_exact30_disjoint_val_cases_and_actual_model_before_jobs(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',parallelism=8))
    protocol=rt.read_bound(prepared['protocol'])
    assert protocol['model']==model and protocol['cases']==cases and protocol['fixed_denominator']==30
    assert protocol['selection_validation_used'] is True and protocol['independent_test_claim'] is False
    assert set(protocol['train_subject_ids']).isdisjoint(c['subject_id'] for c in cases)
    from sim.ausc_policy import isaac_batch
    for arm in ('learned','expert'):
        batch=rt.read_bound(prepared[arm+'_batch'])
        jobs=isaac_batch.validate_jobs(batch)
        assert len(jobs)==30 and batch['parallelism']==8
        assert batch['resource_profile']=='wam_l40s_diagnostic_eight_v1'
        assert all(job['validation_scenario']['arm']==arm for job in jobs)
        command,kw=isaac_batch.child_launch(jobs[0],tmp_path/'out')
        assert kw['start_new_session'] is False and kw['env']['AUSC_VAL_SCENARIO_SEED']=='0'
        assert command[0]==m.NATIVE_PYTHON
    first_job=rt.read_bound(prepared['learned_batch'])['jobs'][0]
    learned=rt.read_bound(dict(path=first_job['config'],sha256=first_job['sha256']))
    binding=m.preflight(learned)
    assert binding['case']==cases[0]
    assert learned['physics']==rt.read_bound(cases[0]['config'])['physics']
    assert 'raw_multimodal_capture' not in learned['fresh_expert']
    assert m.ROUTE_KEY in learned and expanded.ROUTE_KEY not in learned
    from sim.ausc_policy import scene_store
    assert m.ROUTE_KEY not in scene_store._authoring(learned)


def test_predeclared_fresh2_are_absent_from_both_optimizer_and_checkpoint_selection(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'strata'))
    protocol=rt.read_bound(prepared['protocol'])
    assert protocol['strata']['independent_subject_holdout']['subject_ids']==['p000486','p001139']
    assert protocol['strata']['independent_subject_holdout']['scenario_count']==6
    assert protocol['strata']['independent_subject_holdout']['tuning_permitted'] is False
    assert protocol['strata']['checkpoint_selection_validation']['scenario_count']==24
    complete['scope']['validation']['subject_ids'].append('p000486')
    monkeypatch.setattr(expanded,'model_binding',lambda ref:(model,complete))
    with pytest.raises(ValueError,match='selection'):
        m.prepare(model['complete']['path'],source['path'],tmp_path/'selection-leak')
    assert not (tmp_path/'selection-leak').exists()


@pytest.mark.parametrize('fault',['test','train','count','duplicate','seed','physics','changed_source'])
def test_bad_scenario_scope_or_sources_rejected_before_freeze(tmp_path,monkeypatch,fault):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    value=rt.read_bound(source)
    if fault=='test':value['cases'][0]['split']='test'
    if fault=='train':complete['scope']['train']['subject_ids'].append(cases[0]['subject_id']);monkeypatch.setattr(
        expanded,'model_binding',lambda ref:(model,complete))
    if fault=='count':value['cases'].pop()
    if fault=='duplicate':value['cases'][1]=deepcopy(value['cases'][0])
    if fault=='seed':value['cases'][0]['seed']=True
    if fault=='physics':
        original=rt.read_bound(cases[0]['config']);original['physics']['position_iterations']=32
        value['cases'][0]['config']=rt.write_new(tmp_path/'bad-physics.json',original)
    if fault=='changed_source':
        with open(cases[0]['config']['path'],'a') as stream:stream.write(' ')
    altered=rt.write_new(tmp_path/'altered.json',value)
    with pytest.raises(ValueError):m.prepare(model['complete']['path'],altered['path'],tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_new_route_protocol_native_dispatch_and_default_batch_cap(tmp_path,monkeypatch):
    m=module()
    from sim.ausc_policy import fresh_expert,isaac_batch
    from test_ausc_policy_isaac_batch import config
    marker=object();monkeypatch.setattr(m,'run_native_probe',lambda *a:marker)
    assert fresh_expert.run_probe(None,None,None,{m.ROUTE_KEY:{}},None) is marker
    assert m.PROTOCOL_SCHEMA!=expanded.PROTOCOL_SCHEMA
    original=config(tmp_path)
    with pytest.raises(ValueError):isaac_batch.validate_jobs(dict(original,parallelism=5))
    good=dict(original,parallelism=8,resource_profile='wam_l40s_diagnostic_eight_v1',
        diagnostic_only=True,formal_acceptance=False)
    assert len(isaac_batch.validate_jobs(good))==1
    for changes in ({'parallelism':9},{'diagnostic_only':False},{'formal_acceptance':True},
                    {'resource_profile':'unknown'},{'parallelism':True}):
        with pytest.raises(ValueError):isaac_batch.validate_jobs(dict(good,**changes))


def test_paired_seed_and_stratified_fixed_denominator_keep_missing_and_failed_episodes(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared_ref=m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared')
    prepared=rt.read_bound(prepared_ref)
    from sim.ausc_policy import isaac_batch
    jobs=isaac_batch.validate_jobs(rt.read_bound(prepared['expert_batch']))
    command,launch=isaac_batch.child_launch(jobs[0],tmp_path/cases[0]['scenario_id'])
    for key,value in launch['env'].items():
        if key.startswith('AUSC_VAL_SCENARIO_') or key=='AUSC_OWNED_RUN':monkeypatch.setenv(key,value)
    actual=m.seed_worker(jobs[0]['settings'],output=tmp_path/cases[0]['scenario_id'])
    assert actual['seed']==0 and actual['arm']=='expert'
    with pytest.raises(ValueError):m.seed_worker(jobs[0]['settings'],output=tmp_path/'wrong')
    for arm in ('learned','expert'):
        root=tmp_path/arm;root.mkdir()
        batch=rt.read_bound(prepared[arm+'_batch'])
        for i in (0,3):
            case=cases[i];folder=root/case['scenario_id'];folder.mkdir()
            report=dict(physics_steps=12,success=i==0,physical_validation=i==0,errors=[],
                learned_policy=arm=='learned',expert_l2_fallback=False,validation_protocol=prepared['protocol'],
                scenario_id=case['scenario_id'],diagnostic_closed_loop_evaluated=arm=='learned')
            rt.write_new(folder/'fresh_expert_result.json',report)
            rt.write_new(root/(case['scenario_id']+'.result.json'),dict(exit_code=0,config_sha256=batch['jobs'][i]['sha256']))
    result=rt.read_bound(m.summarize(prepared_ref['path'],tmp_path/'learned',tmp_path/'expert',tmp_path/'outcomes.json'))
    assert len(result['outcomes'])==result['fixed_denominator']==30
    assert result['actual_learned_episodes']==2 and result['learned_physical_successes']==1
    assert result['strata']['independent_subject_holdout']['scenario_count']==6
    assert result['strata']['independent_subject_holdout']['learned']['actual_episodes']==1
    assert result['strata']['checkpoint_selection_validation']['scenario_count']==24
    assert result['independent_sealed_test'] is False


def test_wire_request_binds_new_protocol_and_frozen_scenario_without_old_dataset_episode_requirement(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared'))
    first=rt.read_bound(prepared['learned_batch'])['jobs'][0]
    config=rt.read_bound(dict(path=first['config'],sha256=first['sha256']));dep=config[m.ROUTE_KEY]
    ref=rt.pose.require_route(dep['metadata']);q=np.zeros(7)
    state=rt.shared.contract.encode_state(q,ref.pose_base,1.,metadata=dep['metadata'])
    request=m.make_request(dep,1,q,ref.pose_base,1.,metadata=dep['metadata'],history=np.stack([state,state]),history_times=[0.,.05],
        context=dict(observed_at_s=.05,physics_row=12,context=[dep['episode_id'],cases[0]['target'],0,'FREE'],source_id='given-contact-'+ref.sha256))
    m.validate_request(request,dep,sequence=1)
    assert request['binding']['scenario_protocol_sha256']==prepared['protocol']['sha256']
    assert request['binding']['scenario_id']==cases[0]['scenario_id']
    changed=deepcopy(request);changed['binding']['scenario_id']='different-case'
    with pytest.raises(ValueError):m.validate_request(changed,dep,sequence=1)
    with pytest.raises(ValueError):expanded.validate_request(request,dep,sequence=1)
    changed=deepcopy(config);changed[m.ROUTE_KEY]['metadata']['split']='test'
    with pytest.raises(ValueError,match='VALIDATION'):m.preflight(changed)
    changed=deepcopy(config);changed['physics']['position_iterations']=32
    with pytest.raises(ValueError,match='binding'):m.preflight(changed)
