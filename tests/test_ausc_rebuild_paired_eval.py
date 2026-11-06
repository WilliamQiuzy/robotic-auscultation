"""CPU-only protocol checks; no selected policy, native runtime or model calls."""
from copy import deepcopy
import importlib
import importlib.util
import json

import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as expanded
from test_ausc_given_contact_val_scenarios import source_config


def module():
    name='sim.ausc_policy.m2_rebuild_paired_eval'
    assert importlib.util.find_spec(name) is not None, 'New matched-scene adapter required'
    return importlib.import_module(name)


def setup(tmp_path,monkeypatch):
    m=module();cases=[]
    for cohort,subjects in [('frozen_evaluation',m.COHORTS['frozen_evaluation'])]:
        for subject in subjects:
            for pose_id in m.POSES:
                for target in ('L1','R1'):
                    index=len(cases)+1;ref,episode=source_config(tmp_path/str(index),subject,target,index)
                    config=rt.read_bound(ref);split='train' if cohort=='train' else 'validation'
                    meta=config['m2_given_contact_control']['metadata'];meta['split']=split;meta['pose_id']=pose_id
                    # Rebind patient and plan split through the existing authentic fixture.
                    from test_ausc_given_contact_pose import inputs,metadata
                    import numpy as np
                    rawpoint,plan,patient=inputs(tmp_path/str(index))
                    patient_value=rt.pose.bound_json(patient);patient_value.update(identity=subject,split=split,proposal=subject+'_'+pose_id)
                    patient_value['targets'][0]['name']=target
                    patient_ref=rt.write_new(tmp_path/str(index)/'paired-patient.json',patient_value)
                    patient=dict(file=patient_ref['path'],sha256=patient_ref['sha256'])
                    plan_value=rt.pose.bound_json(plan);plan_value.update(name=target,
                        patient_provenance=dict(identity=subject,split=split,asset_sha256='a'*64,binding=patient))
                    plan_ref=rt.write_new(tmp_path/str(index)/'paired-plan.json',plan_value)
                    plan=dict(file=plan_ref['path'],sha256=plan_ref['sha256'])
                    rawpoint.update(name=target,manifest_ref=patient)
                    reference=rt.pose.build_given_reference(rawpoint,plan,patient,np.eye(4),0.)
                    meta.update(metadata(reference),subject_id=subject,split=split,target_id=target,
                        episode_id=episode,pose_id=pose_id,handoff_policy_id=rt.handoff.POLICY_ID)
                    config['fresh_expert']['plans']=[plan]
                    config['m2_given_contact_control'].update(schema=rt.pose.TASK_ID,handoff_policy_id=rt.handoff.POLICY_ID)
                    ref=rt.write_new(tmp_path/str(index)/'paired-config.json',config)
                    cases.append(dict(index=index,scenario_id='case-'+str(index),subject_id=subject,cohort=cohort,
                        split=split,pose_id=pose_id,target_id=target,episode_id=episode,config=ref,seed=0,
                        preparation_status='prepared',reason=None))
    source=rt.write_new(tmp_path/'cases.json',dict(schema=m.INPUT_SCHEMA,complete=True,
        target_ids=['L1','R1'],pose_ids=list(m.POSES),fixed_denominator=len(cases),
        model_outcomes_consulted=False,cases=cases))
    weights=tmp_path/'best.pt';weights.write_bytes(b'new-selected-weights')
    complete_ref=rt.write_new(tmp_path/'ACT_COMPLETE.json',{'new':True})
    model=dict(checkpoint=rt.reference(weights),complete=complete_ref)
    complete=dict(scope={split:dict(subject_ids=list(m.COHORTS[cohort]),episode_ids=['actual-trained-'+cohort])
        for split,cohort in [('train','train'),('validation','selection_validation')]})
    monkeypatch.setattr(expanded,'model_binding',lambda ref:(deepcopy(model),deepcopy(complete)))
    return m,source,model,complete,cases


def test_explicit_two_pose_three_arm_protocol_preserves_scope_and_scene(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',
        astra_settings=m.DEFAULT_ASTRA))
    protocol=rt.read_bound(prepared['protocol'])
    assert protocol['fixed_denominator']==8 and protocol['planned_arm_denominator']==24
    assert protocol['checkpoint_scope']==complete['scope']
    assert protocol['strata']['frozen_evaluation']['subject_ids']==list(m.COHORTS['frozen_evaluation'])
    from sim.ausc_policy import isaac_batch,scene_store
    episodes=set()
    for arm in m.ARMS:
        batch=rt.read_bound(prepared['batches'][arm]);jobs=isaac_batch.validate_jobs(batch)
        assert len(jobs)==8 and batch['parallelism']==1
        config=jobs[0]['settings'];binding=m.preflight_arm(config)
        assert binding['case']==cases[0] and binding['arm']==arm
        assert config['physics']==rt.read_bound(cases[0]['config'])['physics']
        assert 'raw_multimodal_capture' not in config['fresh_expert']
        assert config['matched_scene_eval']['training_eligible'] is False
        assert config['matched_scene_eval']['learned_policy'] is (arm=='act')
        meta=config[m.ROUTE_KEY]['metadata'] if arm=='act' else config['m2_given_contact_control']['metadata']
        episodes.add(meta['episode_id'])
        projected=rt.read_bound(cases[0]['config']);projected.pop('m2_given_contact_control')
        assert scene_store._authoring(config)==scene_store._authoring(projected)
        _,launch=isaac_batch.child_launch(jobs[0],tmp_path/'out')
        assert launch['start_new_session'] is False
    assert len(episodes)==3


@pytest.mark.parametrize('fault',['selection_leak','train_scope','missing_case','duplicate_pose','target_subset','changed_config'])
def test_scope_and_fixed_case_admission_fail_closed_before_output(tmp_path,monkeypatch,fault):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch);value=rt.read_bound(source)
    if fault=='selection_leak':complete['scope']['validation']['subject_ids'].append('p001030')
    if fault=='train_scope':complete['scope']['train']['subject_ids'].pop()
    if fault=='missing_case':value['cases'].pop()
    if fault=='duplicate_pose':value['cases'][1]['pose_id']=value['cases'][2]['pose_id']
    if fault=='target_subset':value['target_ids']=['L1']
    if fault=='changed_config':open(cases[0]['config']['path'],'a').write(' ')
    monkeypatch.setattr(expanded,'model_binding',lambda ref:(model,complete))
    source=rt.write_new(tmp_path/'altered.json',value)
    with pytest.raises(ValueError):m.prepare(model['complete']['path'],source['path'],tmp_path/'rejected',astra_settings=m.DEFAULT_ASTRA)
    assert not (tmp_path/'rejected').exists()


def test_missing_failed_and_software_pass_only_stay_in_all_denominators(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    value=rt.read_bound(source);value['cases'][-1].update(config=None,preparation_status='failed',reason='source unavailable')
    source=rt.write_new(tmp_path/'missing-source.json',value)
    ref=m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',astra_settings=m.DEFAULT_ASTRA)
    prepared=rt.read_bound(ref);roots={arm:tmp_path/arm for arm in m.ARMS}
    for arm,root in roots.items():
        root.mkdir();batch=rt.read_bound(prepared['batches'][arm]);job=batch['jobs'][0]
        folder=root/job['id'];folder.mkdir()
        report=rt.write_new(folder/'fresh_expert_result.json',dict(success=True,physical_validation=True,physics_steps=240,errors=[]))
        rt.write_new(root/(job['id']+'.result.json'),dict(exit_code=0,config_sha256=job['sha256'],result_sha256=report['sha256']))
    summary=rt.read_bound(m.summarize(ref['path'],roots,tmp_path/'summary.json'))
    assert len(summary['outcomes'])==8 and summary['planned_arm_denominator']==24
    for arm in m.ARMS:
        assert summary['arms'][arm]['qualified_physical_successes']==0
        assert summary['outcomes'][0]['arms'][arm]['status']=='unqualified_native_result'
        assert summary['outcomes'][-1]['arms'][arm]['status']=='preparation_failure'
        assert summary['arms'][arm]['planned_cases']==8
        assert summary['arms'][arm]['qualified_physical_success_rate']==0.
    assert summary['qualified_physical_result_exists'] is False


def test_new_act_route_dispatches_and_audit_accepts_only_explicit_adapter(tmp_path,monkeypatch):
    m=module()
    from sim.ausc_policy import fresh_expert,m2_given_contact_pilot_native_audit as audit
    marker=object();monkeypatch.setattr(m,'run_native_probe',lambda *a:marker)
    assert fresh_expert.run_probe(None,None,None,{m.ROUTE_KEY:{}},None) is marker
    assert audit.adapter_for({m.ROUTE_KEY:{}}) is m
    assert m.PROTOCOL_SCHEMA!=expanded.PROTOCOL_SCHEMA


@pytest.mark.parametrize('arm',['expert','astra'])
def test_neutral_analytic_observer_keeps_full_actual_ticks_without_forging_act(tmp_path,monkeypatch,arm):
    from types import SimpleNamespace
    import numpy as np
    from sim.ausc_policy.m2_given_contact_controller import GivenContactExpert
    from sim.ausc_policy.astra_native import AstraGivenContactExpert
    from sim.ausc_policy import m2_given_contact_evaluation as neutral
    from sim.ausc_policy import m2_given_contact_pilot_native_audit as audit
    m=module();kind=AstraGivenContactExpert if arm=='astra' else GivenContactExpert
    c=kind.__new__(kind);state=dict(time_s=0.,q=np.zeros(7),tcp_base=np.array([0.,0.,.1]),rotation_world=np.eye(3))
    c.env=SimpleNamespace(rows=[],base=np.eye(4),state=lambda:deepcopy(state))
    c.metadata=dict(reference_sha256='a'*64,episode_id='analytic-test');c.reference=SimpleNamespace(sha256='a'*64)
    c.episode_id='analytic-test';c.trajectory=None;c.control_owner='PREPARATION';c.actor_phase=None
    c.manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='FREE')))
    c._current_intent_audit=None;c.q=np.zeros(7)
    def tick(command,subphase):
        state['time_s']+=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=state['q'].tolist(),q_command=list(command),
            phase='FREE',subphase=subphase,physical_violations=[],diaphragm_N=4.,
            command_source_id='original_analytic_motor',retainer_front_audit={'actual':True}))
    c._tick=tick
    source=tmp_path/'source.py';source.write_text('# immutable test source\n')
    inventory={'source':rt.reference(source)};config={'bound':'synthetic-observer-test',m.DESCRIPTOR:dict(protocol=rt.reference(source))}
    binding=dict(arm=arm,metadata=deepcopy(c.metadata),case={'config':rt.reference(source),'scenario_id':'analytic-test'})
    monkeypatch.setattr(m,'preflight_arm',lambda *a,**k:deepcopy(binding));monkeypatch.setattr(m,'sources',lambda:inventory)
    observer=m.attach_analytic(c,config,tmp_path)
    for _ in range(241):c._tick([0.]*7,'staging')
    receipt=neutral.close_native_audit(c)
    assert receipt['complete'] is True and observer.stream.batch_flushes==2 and audit.installed(c)
    assert receipt['pilot_binding']['model'] is None and receipt['pilot_binding']['arm']==arm
    assert receipt['pilot_binding']['supervisor_model']==('gpt-6-astra' if arm=='astra' else None)
    events=neutral.read_native_trace(receipt,c.metadata)
    assert len(events)==241 and all(e['native_row']['command_source_id']=='original_analytic_motor' for e in events)
    assert all(e['actual_control_owner']=='PREPARATION' for e in events)
    assert c._tick is tick
    # A software-success report with an intact trace still reaches the original
    # gate and fails: this synthetic episode has no physical HOLD/plane evidence.
    run=dict(success=True,physical_validation=True,errors=[],learned_policy=False,physics_steps=241,
        neutral_native_audit=receipt,trace_file=receipt['trace']['path'],trace_sha256=receipt['trace']['sha256'],
        runtime_sources_sha256={str(source):rt.reference(source)['sha256']})
    if arm=='expert':
        binding['protocol']={};binding['case']['config']=rt.reference(source)
        rt.write_new(tmp_path/'fresh_expert_result.json',run);rt.write_new(tmp_path/'COMPLETE.json',run)
        called=[];physical=rt.dataset._physical
        def unchanged_gate(*args):called.append(True);return physical(*args)
        monkeypatch.setattr(rt.dataset,'_physical',unchanged_gate)
        value=m.qualification_value(tmp_path,config)
        assert called and value['passed'] is False and value['checks'].get('unchanged_physical_gate_passed') is not True
        # Even if dataset scalar gates pass, the independent original native
        # command/full-pose/home recheck is mandatory before acceptance.
        monkeypatch.setattr(rt.dataset,'_physical',lambda *a:None)
        extra=[]
        def original_native_gate(*args):
            extra.append(True);raise ValueError('original native command/home gate rejected')
        monkeypatch.setattr(neutral,'_physical_native',original_native_gate)
        value=m.qualification_value(tmp_path,config)
        assert extra and value['passed'] is False and 'command/home gate' in value['error']


def test_bad_astra_endpoint_fails_before_any_preparation(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Loopback'):
        m.prepare(model['complete']['path'],source['path'],tmp_path/'bad-endpoint',
            astra_settings=dict(m.DEFAULT_ASTRA,endpoint='http://example.com/decision'))
    assert not (tmp_path/'bad-endpoint').exists()


def test_reported_ticks_without_neutral_trace_are_not_actual_native_evidence(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',astra_settings=m.DEFAULT_ASTRA))
    root=tmp_path/'expert';root.mkdir();job=rt.read_bound(prepared['batches']['expert'])['jobs'][0]
    folder=root/job['id'];folder.mkdir()
    result=rt.write_new(folder/'fresh_expert_result.json',dict(success=False,physical_validation=False,physics_steps=999,
        failure_reason='unverified reported tick count',errors=[]))
    rt.write_new(root/(job['id']+'.result.json'),dict(exit_code=0,config_sha256=job['sha256'],result_sha256=result['sha256']))
    value=m._outcome(cases[0],'expert',root,job,prepared['protocol'])
    assert value['reported_physics_steps']==999
    assert value['actual_native_episode'] is False and value['physics_steps']==0


@pytest.mark.parametrize('arm',['expert','astra'])
def test_shared_scene_accepts_authenticated_paired_analytic_without_training_capture(tmp_path,monkeypatch,arm):
    from test_ausc_policy_scene_store import bound
    from sim.ausc_policy import scene_store
    m=module();binding,config=bound(tmp_path)
    config['fresh_expert'].pop('raw_multimodal_capture');config[m.DESCRIPTOR]={'arm':arm}
    calls=[]
    monkeypatch.setattr(m,'preflight_arm',lambda value,**k:calls.append(value[m.DESCRIPTOR]['arm']) or {})
    assert scene_store.validate_config(config)[0]==binding
    assert calls
    config['physics']['gravity']=0.
    with pytest.raises(ValueError,match='authoring'):scene_store.validate_config(config)


def test_original_9000_second_source_budget_is_retained_for_all_matched_arms(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch);value=rt.read_bound(source)
    for index,case in enumerate(value['cases']):
        config=rt.read_bound(case['config']);config['time_limit_seconds']=9000
        case['config']=rt.write_new(tmp_path/('original-budget-'+str(index)+'.json'),config)
    source=rt.write_new(tmp_path/'original-budget-cases.json',value)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',astra_settings=m.DEFAULT_ASTRA))
    from sim.ausc_policy import isaac_batch,scene_store
    assert m.MAX_CASE_SECONDS==9000
    original=rt.read_bound(value['cases'][0]['config']);original.pop('m2_given_contact_control')
    for arm in m.ARMS:
        batch=rt.read_bound(prepared['batches'][arm]);jobs=isaac_batch.validate_jobs(batch)
        assert all(job['settings']['time_limit_seconds']==9000 for job in jobs)
        assert scene_store._authoring(jobs[0]['settings'])==scene_store._authoring(original)
        assert jobs[0]['settings']['physics']==original['physics']


@pytest.mark.parametrize('seconds',[9001,1000000,None,True,60,float('inf'),9000.])
def test_case_budget_still_requires_finite_bounded_integer(tmp_path,monkeypatch,seconds):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    config=rt.read_bound(cases[0]['config']);config['time_limit_seconds']=seconds
    with pytest.raises(ValueError,match='bounded'):
        m.validate_expert_config(config,cases[0])


def test_astra_serial_resource_binding_preserves_all_scene_physics(tmp_path,monkeypatch):
    m,source,model,complete,cases=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',astra_settings=m.DEFAULT_ASTRA,parallelism=3))
    protocol=rt.read_bound(prepared['protocol'])
    expected=dict(expert=3,act=3,astra=1)
    assert prepared['arm_parallelism']==protocol['arm_parallelism']==expected
    from sim.ausc_policy import isaac_batch
    for arm in m.ARMS:
        batch=rt.read_bound(prepared['batches'][arm]);jobs=isaac_batch.validate_jobs(batch)
        assert batch['parallelism']==expected[arm]
        assert jobs[0]['settings']['physics']==rt.read_bound(cases[0]['config'])['physics']
        batch['parallelism']=2
        with pytest.raises(ValueError,match='parallelism'):isaac_batch.validate_jobs(batch)
