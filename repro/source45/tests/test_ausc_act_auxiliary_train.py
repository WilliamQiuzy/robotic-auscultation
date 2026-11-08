"""Mixed-source optimization is explicit, valid-cell weighted and provenance bound."""
from copy import deepcopy
import importlib
import importlib.util

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt


def trainer():
    name='sim.ausc_policy.m2_act_auxiliary_train'
    assert importlib.util.find_spec(name) is not None, 'Explicit auxiliary optimization trainer required'
    return importlib.import_module(name)


def scope_value():
    from sim.ausc_policy import m2_act_recovery_scope as scope
    return dict(schema='wam_act_recovery_train16_scope_v2',
        cohorts=dict(train=list(scope.TRAIN16_IDS),selection_validation=['p000923'],
            frozen_evaluation=list(scope.FRESH8_IDS)),
        optimized_train_ids=sorted(scope.TRAIN16_IDS[:5]),unavailable_train_ids=sorted(scope.TRAIN16_IDS[5:]),
        parent_cohort=dict(path='/opaque/COHORT_FREEZE.json',bytes=100,sha256=scope.PARENT_COHORT_SHA256),
        pose_ids=['p000474_a001230_f0000','p000476_a001231_f0023'],target_ids=['R3','L3'],
        fixed_denominator=32,frozen_before_training=True,model_outcomes_consulted=False,
        development_outcomes_used_for_repair=True,fresh_evaluation_outcomes_consulted=False,
        fresh_payloads_opened=False,timestamp_utc='2026-11-06T21:00:00Z')


def test_train16_scope_declares_pool_actual_ids_and_opaque_same_fresh_cohort():
    from sim.ausc_policy import m2_act_recovery_scope as scope
    value=scope_value()
    got=scope.validate_train16_metadata(value,actual_train_ids=value['optimized_train_ids'])
    assert got['train']==list(scope.TRAIN16_IDS)
    assert len(got['frozen_evaluation'])==8
    assert scope.SCHEMA=='wam_act_recovery_new_identity_cohort_v1'


@pytest.mark.parametrize('fault',['val_overlap','fresh_overlap','invented_actual','hidden_unavailable','actual_mismatch','outcomes'])
def test_scope_contamination_and_inflated_optimization_inventory_rejected(fault):
    from sim.ausc_policy import m2_act_recovery_scope as scope
    value=scope_value();actual=value['optimized_train_ids'].copy()
    if fault=='val_overlap':value['optimized_train_ids'].append('p000923')
    elif fault=='fresh_overlap':value['optimized_train_ids'].append(scope.FRESH8_IDS[0])
    elif fault=='invented_actual':value['optimized_train_ids'].append('p009999')
    elif fault=='hidden_unavailable':value['unavailable_train_ids'].pop()
    elif fault=='actual_mismatch':actual.pop()
    else:value['model_outcomes_consulted']=True
    with pytest.raises(ValueError):scope.validate_train16_metadata(value,actual_train_ids=actual)


def test_new_scope_authenticates_parent_metadata_without_reading_fresh_payloads(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_act_recovery_scope as scope
    value=scope_value();ref=rt.write_new(tmp_path/'new-scope.json',value);original_read=rt.read_bound;parents=[]
    parent=dict(cohorts=dict(frozen_evaluation=value['cohorts']['frozen_evaluation']),
        pose_ids=value['pose_ids'],target_ids=value['target_ids'])
    monkeypatch.setattr(rt,'read_bound',lambda item:parent if item==value['parent_cohort'] else original_read(item))
    def legacy(item):parents.append(item);return parent['cohorts']
    monkeypatch.setattr(scope,'_validate_legacy',legacy)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Fresh numeric payload opened'))
    assert scope.validate(ref)==value['cohorts'] and parents==[value['parent_cohort']]


def source(kind,subject='p001120',pose='p000474_a001230_f0000',target='R3'):
    from sim.ausc_policy import m2_act_auxiliary_data as data
    result=dict(schema=data.SOURCE_SCHEMA,kind=kind,subject_id=subject,split='train',pose_id=pose,target_id=target,
        metadata=dict(subject_id=subject,split='train'),**data.FLAGS)
    if kind=='planned_ideal_tracking':result.update(verification_profile='original_full_cpu_preview',injected_factories=[])
    else:result.update(source_episode_verdict_unchanged=True,original_source_success=False,
        observation_origin='actual_authenticated_TRAIN_student_query',bindings={k:{} for k in
            ('generator','native','owner','child','proposal','outcome','request','diagnostic','diagnostic_root',
             'frozen_request','frozen_manifest','original_arrays','source','geometry')})
    return result


@pytest.mark.parametrize('fault',['fixture','injection','val','fresh','native_claim','bad_oracle'])
def test_auxiliary_metadata_rejected_before_any_numerical_loader(tmp_path,monkeypatch,fault):
    m=trainer();from sim.ausc_policy import m2_act_auxiliary_data as data
    sources=[source(data.KINDS[0]),source(data.KINDS[1])]
    if fault=='fixture':sources[0]['verification_profile']='injected_cpu_fixture_unqualified'
    elif fault=='injection':sources[0]['injected_factories']=['trajectory_factory']
    elif fault=='val':sources[1]['metadata']['split']='validation'
    elif fault=='fresh':sources[0]['subject_id']='p000519'
    elif fault=='native_claim':sources[0]['physics_qualified']=True
    else:sources[1].pop('original_source_success')
    metadata=rt.write_new(tmp_path/'metadata.json',dict(sources=sources))
    complete=rt.write_new(tmp_path/'complete.json',dict(schema=data.SCHEMA,complete=True,metadata=metadata,**data.FLAGS))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Numerical archive opened before eligibility gate'))
    with pytest.raises(ValueError):m.auxiliary_metadata(complete)


def test_each_group_official_valid_mean_gets_exact_objective_weight_and_padding_is_ignored():
    m=trainer();calls=[]
    class OfficialForward:
        def __call__(self,batch):
            calls.append(batch)
            valid=~batch['action_is_pad']
            return (np.abs(batch['action'])*valid[...,None]).sum()/(valid.sum()*7),{}
    batches={}
    for name,count,error,horizon in [('native',32,1.,20),('planned',16,2.,7),('oracle',16,3.,1)]:
        mask=np.tile(np.arange(20)>=horizon,(count,1));actions=np.full((count,20,7),error)
        actions[mask]=123456.
        batches[name]={'action':actions,'action_is_pad':mask}
    loss,logged=m.mixed_loss(OfficialForward(),batches)
    assert loss==pytest.approx(1.75) and len(calls)==3
    assert [logged[k]['coefficient'] for k in batches]==[.5,.25,.25]
    assert logged['oracle']['valid_cells']==16 and logged['oracle']['valid_joints']==112
    batches['oracle']['action'][batches['oracle']['action_is_pad']]=-987654.
    again,_=m.mixed_loss(OfficialForward(),batches)
    assert again==loss


def cache_inputs(tmp_path):
    from test_ausc_given_contact_expanded_pilot import bundle
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as old
    prepared=old.load_dataset(rt.write_new(tmp_path/'native.json',bundle(tmp_path)))
    sources=[source('planned_ideal_tracking',subject='p001120'),source('planned_ideal_tracking',subject='p000758'),
        source('student_oracle_unexecuted',subject='p001120')]
    n=44;histories=np.arange(n*34).reshape(n,2,17)/1000
    samples=dict(source_histories=histories,relative_setters=np.ones((n,20,7)),
        setter_is_pad=np.tile(np.arange(20)>=1,(n,1)),source_indices=np.r_[np.zeros(40,int),np.ones(2,int),np.full(2,2)])
    return prepared,dict(samples=samples,sources=sources,counts={})


def test_cache_packs_once_reuses_real_stats_and_samples_body_before_windows(tmp_path,monkeypatch):
    m=trainer();prepared,aux=cache_inputs(tmp_path)
    from sim.ausc_policy import m2_act_auxiliary_data as data
    packed_calls=[]
    def pack(samples,stats,ids):
        packed_calls.append((stats,ids.copy()))
        h=(samples['source_histories'][ids]-np.array(stats['state_mean']))/np.array(stats['state_std'])
        return {'observation.state':h[:,1,:7].astype(np.float32),
            'observation.environment_state':np.concatenate((h[:,1,7:],h[:,0]),axis=1).astype(np.float32),
            'action':((samples['relative_setters'][ids]-stats['action_mean'])/stats['action_std']).astype(np.float32),
            'action_is_pad':samples['setter_is_pad'][ids]}
    monkeypatch.setattr(data,'pack_batch',pack)
    cache=m.build_cache(prepared,aux)
    monkeypatch.setattr(data,'validate_samples',lambda *a:pytest.fail('Full validator in optimization loop'))
    monkeypatch.setattr(data,'pack_batch',lambda *a:pytest.fail('Repacking inside optimization loop'))
    rng=np.random.default_rng(42);observed=[]
    for _ in range(100):
        ids=m.sample_cached_indices(rng,cache)
        assert {k:len(v) for k,v in ids.items()}==dict(native=32,planned=16,oracle=16)
        assert np.all(np.isin(ids['oracle'],[42,43]))
        observed.extend(ids['planned'])
        batches=m.cached_batches(cache,ids)
        assert batches['native']['action'].shape==(32,20,7)
        assert batches['oracle']['observation.environment_state'].shape==(16,27)
    assert .44<np.mean(np.array(observed)>=40)<.56
    assert len(packed_calls)==1 and packed_calls[0][0] is prepared['normalization']
    np.testing.assert_array_equal(packed_calls[0][1],np.arange(44))


def test_cache_refuses_missing_auxiliary_kind(tmp_path,monkeypatch):
    m=trainer();prepared,aux=cache_inputs(tmp_path)
    aux['sources']=aux['sources'][:2]
    with pytest.raises(ValueError,match='both|Both|oracle'):m.build_cache(prepared,aux)


def test_explicit_new_dispatch_preserves_legacy_schedule_and_schema(monkeypatch):
    m=trainer();from sim.ausc_policy import m2_given_contact_expanded_pilot_train as old
    assert old.PILOT_SCHEMA=='m2_given_contact_expanded_act_pilot_v1'
    config={'schema':m.CONFIG_SCHEMA};sentinel=object()
    monkeypatch.setattr(m,'validate_config',lambda value:sentinel)
    assert old.validate_config(config) is sentinel
    monkeypatch.setattr(m,'training_protocol',lambda value:sentinel)
    assert old.training_protocol(config) is sentinel
    monkeypatch.setattr(m,'load_selected_policy',lambda value:sentinel)
    assert old.load_selected_policy({'schema':m.SELECTION_SCHEMA}) is sentinel
    assert old.SCHEDULE['steps']==20000 and old.SCHEDULE['batch_size']==64


def test_new_model_admission_does_not_relabel_train16_as_legacy_train4(monkeypatch):
    m=trainer();from sim.ausc_policy import m2_rebuild_paired_eval as paired
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment
    value=scope_value();optimized=value['optimized_train_ids']
    complete=dict(schema=m.PILOT_SCHEMA,scope=dict(train=dict(subject_ids=optimized),
        validation=dict(subject_ids=['p000923'])),optimization_scope=value)
    monkeypatch.setattr(deployment,'model_binding',lambda ref:({},complete))
    assert paired.model_binding({})[1] is complete
    complete['scope']['train']['subject_ids'].append('p000519')
    with pytest.raises(ValueError):paired.model_binding({})


def test_new_paired_protocol_requires_pretraining_scope_or_exact_unchanged_parent():
    m=trainer();from sim.ausc_policy import m2_rebuild_paired_eval as paired
    frozen=scope_value();new_ref={'path':'/new/scope.json','bytes':1,'sha256':'a'*64}
    complete=dict(schema=m.PILOT_SCHEMA,optimization_scope=frozen,schedule=dict(evaluation_cohort=new_ref))
    for ref in (new_ref,frozen['parent_cohort']):paired._require_training_freeze(dict(identity_cohort=ref),complete)
    with pytest.raises(ValueError):paired._require_training_freeze({},complete)
    with pytest.raises(ValueError):paired._require_training_freeze(dict(identity_cohort=dict(new_ref,sha256='b'*64)),complete)
    paired._require_training_freeze({},dict(schema='m2_given_contact_expanded_act_pilot_v1'))


def test_valid_mean_official_forward_bytes_must_match_pinned_inventory():
    m=trainer()
    ref=dict(path='/vendor/src/lerobot/policies/act/modeling_act.py',bytes=1,sha256=m.OFFICIAL_FORWARD_SHA256)
    software=dict(vendor=dict(files={'src/lerobot/policies/act/modeling_act.py':ref}))
    m.validate_official_forward(software)
    ref['sha256']='0'*64
    with pytest.raises(ValueError,match='valid|official|Official'):m.validate_official_forward(software)


def test_protocol_preserves_original_gradient_clip_and_forbids_action_clipping():
    m=trainer();protocol=m.training_protocol(dict(m.SCHEDULE))
    assert protocol['gradient_clip_norm']==1.0 and protocol['gradient_clip_error_if_nonfinite'] is True
    assert protocol['action_clipping'] is False
    assert protocol['augmentation']==dict(q_noise_std=0.,q_dropout=0.)
    from sim.ausc_policy import rebuild_gpu_binding as binding
    assert protocol['auxiliary_admission']['schema']==binding.AUXILIARY_ADMISSION_SCHEMA


def test_unpinned_oracle_archive_is_rejected_before_running_its_python(tmp_path,monkeypatch):
    m=trainer();s=source('student_oracle_unexecuted')
    s['bindings']['request']=rt.write_new(tmp_path/'REQUEST.json',dict(source={}))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('Unpinned oracle archive executed'))
    with pytest.raises(ValueError,match='source20|Source20|frozen|Frozen'):m.authenticate_oracles(dict(sources=[s]))


@pytest.mark.parametrize('fault',['history','time','first_label','mask'])
def test_resealed_oracle_arrays_must_equal_actual_query_and_bound_first_setter(tmp_path,fault):
    m=trainer();from test_ausc_act_auxiliary_data import oracle_fixture
    from sim.ausc_policy import m2_act_auxiliary_data as data
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    aux=data.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
    samples=aux['samples'];proposal=rt.read_bound(samples['sources'][0]['bindings']['proposal'])
    m.validate_oracle_sample(samples,0,proposal,proposal['query'])
    if fault=='history':samples['source_histories'][0,0,3]+=.01
    elif fault=='time':samples['source_times'][0,0]+=.001
    elif fault=='first_label':samples['relative_setters'][0,0,2]+=.001
    else:samples['setter_is_pad'][0,2]=False
    with pytest.raises(ValueError,match='genuine|first|oracle|Oracle'):m.validate_oracle_sample(samples,0,proposal,proposal['query'])


def test_new_checkpoint_marker_survives_cpu_bridge_and_paired_decode(tmp_path):
    m=trainer();from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment
    from sim.ausc_policy import m2_rebuild_paired_eval as paired
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='new-act',target_id='L1')
    q=np.zeros(7);state=rt.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    dep=dict(metadata=meta,episode_id='new-act',source_episode_id='native',
        model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema=m.PILOT_SCHEMA),
        source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='scenario')
    for adapter in (deployment,paired):
        request=adapter.make_request(dep,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),
            history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,
                context=['new-act','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,dep,sequence=1)
        block=adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request)
        assert block.metadata['checkpoint_policy']==m.PILOT_SCHEMA


@pytest.mark.parametrize('change',[dict(steps=500),dict(eval_every=250),dict(batch_size=32),dict(q_noise_std=.002),
    dict(lr=.01),dict(steps=True),dict(resume='old.pt')])
def test_new_config_cannot_silently_change_full_schedule_before_payloads(change):
    m=trainer()
    config=dict(m.SCHEDULE,schema=m.CONFIG_SCHEMA,base='/workspace/wam',dataset=m.NATIVE_DATASET,
        auxiliary={},evaluation_cohort={});config.update(change)
    with pytest.raises(ValueError,match='schedule'):m.validate_config(config)


def test_public_config_truthfully_admits_separate_train16_headers_without_numerical_reads(tmp_path,monkeypatch):
    m=trainer();from sim.ausc_policy import m2_act_recovery_scope as scope
    from sim.ausc_policy import rebuild_gpu_binding as binding,m2_act_auxiliary_data as data
    frozen=scope_value()
    native=dict(schema='m2_given_contact_masked_preparation_v1',complete=True,rebuild_epoch=binding.EPOCH,
        original_cases=0,diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False,
        task_contract=rt.task_contract(),counts=m.NATIVE_COUNTS,
        split_subject_ids=dict(train=frozen['cohorts']['train'][:4],validation=['p000923']))
    native_ref=rt.write_new(tmp_path/'native-header.json',native)
    monkeypatch.setattr(m,'NATIVE_DATASET',native_ref)
    planned=source(data.KINDS[0],subject='p000961');oracle=source(data.KINDS[1])
    oracle['bindings'].update(frozen_manifest=m.SOURCE20_MANIFEST,generator=m.SOURCE20_GENERATOR,
        frozen_request=dict(sha256=m.SOURCE20_REQUEST_SHA256))
    metadata=rt.write_new(tmp_path/'aux-metadata.json',dict(sources=[planned,oracle]))
    aux_ref=rt.write_new(tmp_path/'aux-header.json',dict(schema=data.SCHEMA,complete=True,metadata=metadata,**data.FLAGS))
    scope_ref=rt.write_new(tmp_path/'scope.json',frozen)
    monkeypatch.setattr(scope,'validate',lambda ref:rt.read_bound(ref)['cohorts'])
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Configuration admission opened numerical arrays'))
    config=dict(m.SCHEDULE,schema=m.CONFIG_SCHEMA,base='/workspace/wam',dataset=native_ref,auxiliary=aux_ref,evaluation_cohort=scope_ref)
    assert m.validate_config(config) is config
    changed=deepcopy(frozen);changed['optimized_train_ids'].remove('p000961')
    changed['unavailable_train_ids']=sorted(set(scope.TRAIN16_IDS)-set(changed['optimized_train_ids']))
    with pytest.raises(ValueError,match='actual contributing'):
        m.validate_config(dict(config,evaluation_cohort=rt.write_new(tmp_path/'false-scope.json',changed)))
    native['split_subject_ids']['train']=list(scope.TRAIN16_IDS)
    with pytest.raises(ValueError,match='TRAIN4'):binding.validate_dataset(native,schema=native['schema'],task_contract=rt.task_contract())


def completed_run(tmp_path,monkeypatch):
    """Bound metadata fixture; never presented as actual inference evidence."""
    m=trainer();value=scope_value();actual=value['optimized_train_ids']
    config=dict(m.SCHEDULE,schema=m.CONFIG_SCHEMA,base='/workspace/wam',dataset=deepcopy(m.NATIVE_DATASET),
        auxiliary=rt.write_new(tmp_path/'aux.json',{}),evaluation_cohort=rt.write_new(tmp_path/'scope.json',value))
    monkeypatch.setattr(m,'validate_config',lambda c:c)
    inputs=dict(config=rt.write_new(tmp_path/'config.json',config),dataset=config['dataset'],auxiliary=config['auxiliary'],
        code=m.code_binding())
    weights=tmp_path/'actual-selected.pt';weights.write_bytes(b'hash-bound test fixture; not actual model weights')
    best=rt.reference(weights)
    metrics={s:rt.metrics(np.zeros((n,20,7)),dict(actions=np.zeros((n,20,7)),
        histories=np.zeros((n,2,17)),states=np.zeros((n,17)),velocities=np.zeros((n,7))))
        for s,n in [('train',2),('validation',2)]}
    for split,count in m.NATIVE_COUNTS.items():metrics[split]['sample_count']=count
    native_scope={s:dict(subject_ids=sorted(ids),episode_ids=[s+'-native'],episode_count=1,window_count=m.NATIVE_COUNTS[s])
        for s,ids in [('train',value['cohorts']['train'][:4]),('validation',['p000923'])]}
    optimization_scope=deepcopy(native_scope);optimization_scope['train'].update(subject_ids=actual,
        native_subject_ids=native_scope['train']['subject_ids'],native_episode_count=1,native_window_count=8373)
    counts={kind:dict(attempted_slots=1,accepted_sources=1,samples=2,supervised_cells=2) for kind in
        ('planned_ideal_tracking','student_oracle_unexecuted')}
    def draws(step):
        def subjects(name):return native_scope['train']['subject_ids'] if name=='native' else actual
        return dict(actual_optimized_train_ids=actual,gradient_examples={k:step*v['size'] for k,v in m.GROUPS.items()},
            gradient_subject_examples={k:{subjects(k)[0]:step*v['size']-len(subjects(k))+1,
                **{i:1 for i in subjects(k)[1:]}} for k,v in m.GROUPS.items()})
    selection=dict(schema=m.SELECTION_SCHEMA,inputs=inputs,best=best,selected_step=500,scope=optimization_scope,
        native_scope=native_scope,optimization_scope=value,auxiliary_counts=counts,
        oracle_authentication=[dict(actual_history_authenticated=True,future_measurements_claimed=False)],
        selected_optimization=draws(500),optimization=draws(20000),protocol=m.training_protocol(config),
        selected_metrics=metrics,window_order=dict(train='a'*64,validation='b'*64),**m.FLAGS)
    comparison=dict(passed=True,prediction_max_abs_rad=0.,prediction_atol=1e-6,prediction_rtol=1e-5,metric_atol=1e-7,metric_rtol=1e-5)
    cpu=rt.write_new(tmp_path/'cpu.json',dict(schema=m.CPU_SCHEMA,passed=True,checkpoint=best,selected_step=500,
        scope=optimization_scope,optimization_scope=value,selection=rt.write_new(tmp_path/'SELECTED.json',selection),
        comparisons={s:deepcopy(comparison) for s in ('train','validation')},metrics=metrics,**m.FLAGS))
    return dict(selection,schema=m.PILOT_SCHEMA,complete=True,completed_steps=20000,schedule=config,
        task_contract=rt.task_contract(),model=rt.shared.native_spec(),cpu_equivalence=cpu,
        evaluations=[dict(step=step,**deepcopy(metrics)) for step in range(500,20001,500)])


def test_new_completion_and_cpu_receipt_bind_actual_optimized_inventory_and_tolerances(tmp_path,monkeypatch):
    m=trainer();from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment
    complete=completed_run(tmp_path,monkeypatch)
    model,loaded=deployment.model_binding(rt.write_new(tmp_path/'COMPLETE.json',complete))
    assert loaded==complete and model['checkpoint_schema']==m.PILOT_SCHEMA
    for fault in ('unknown_actual','lost_native','counts','selected_step','cpu_schema','cpu_tolerance','cpu_scope'):
        changed=deepcopy(complete)
        if fault=='unknown_actual':changed['optimization']['actual_optimized_train_ids'].append('p000519')
        elif fault=='lost_native':changed['scope']['train']['native_window_count']=9000
        elif fault=='counts':changed['selected_optimization']['gradient_examples']['oracle']-=1
        elif fault=='selected_step':changed['selected_step']=1000
        else:
            receipt=rt.read_bound(changed['cpu_equivalence'])
            if fault=='cpu_schema':receipt['schema']='m2_given_contact_expanded_act_cpu_equivalence_v1'
            elif fault=='cpu_tolerance':receipt['comparisons']['validation']['prediction_atol']=1e-2
            else:receipt['optimization_scope']['optimized_train_ids'].append('p000519')
            changed['cpu_equivalence']=rt.write_new(tmp_path/(fault+'-cpu.json'),receipt)
        with pytest.raises(ValueError):deployment.model_binding(rt.write_new(tmp_path/(fault+'.json'),changed))


def test_new_checkpoint_load_uses_exact_cpu_software_and_selected_packet(tmp_path,monkeypatch):
    m=trainer();from types import SimpleNamespace
    complete=completed_run(tmp_path,monkeypatch)
    software=dict(vendor=dict(files={'src/lerobot/policies/act/modeling_act.py':
        dict(sha256=m.OFFICIAL_FORWARD_SHA256)}),dependencies=[],python={},prefix='/official',python_version='x',source_bindings={})
    complete['inputs']['software']=rt.write_new(tmp_path/'software.json',software)
    # Reuse actual normalization contract rather than duplicating its version fields.
    from test_ausc_act_auxiliary_data import fixture
    (tmp_path/'normalization').mkdir()
    samples=fixture(tmp_path/'normalization')
    from sim.ausc_policy import m2_given_contact_act as given
    native_samples=dict(schema=given.MASKED_DATASET_SCHEMA,task_id=given.TASK_ID,
        observation_contract=given.observation_contract(),action_semantics_id=given.ACTION_SEMANTICS_ID,
        states=samples['source_histories'][:,1],histories=samples['source_histories'],actions=samples['relative_setters'],
        action_is_pad=samples['setter_is_pad'],label_protocol=rt.shared.MASKED_LABEL_PROTOCOL)
    normalization=given.fit_normalization(native_samples,split='train')
    monkeypatch.setattr(m,'native_normalization',lambda config:normalization)
    packet={k:deepcopy(complete[k]) for k in ('inputs','scope','protocol','optimization_scope','native_scope',
        'auxiliary_counts','oracle_authentication')}
    packet.update(schema=m.PILOT_SCHEMA,step=500,task_contract=rt.task_contract(),model=rt.shared.native_spec(),
        normalization=deepcopy(normalization),state_dict={'selected':'only'},optimization=complete['selected_optimization'],**m.FLAGS)
    loaded=[]
    policy=SimpleNamespace(load_state_dict=lambda value,strict:loaded.append((value,strict)),eval=lambda:None)
    torch=SimpleNamespace(load=lambda *a,**k:packet)
    monkeypatch.setattr(rt,'make_model',lambda *a,**k:(torch,policy,software))
    assert m.load_selected_policy(complete)[2] is packet
    assert loaded==[({'selected':'only'},True)]
    changed=deepcopy(complete)
    changed['selected_optimization']['actual_optimized_train_ids']=[*changed['selected_optimization']['actual_optimized_train_ids'],'p000519']
    packet['optimization']=changed['selected_optimization']
    with pytest.raises(ValueError,match='optimized|contribut'):m.load_selected_policy(changed)
    packet['optimization']=complete['selected_optimization']
    packet['normalization']['state_mean'][0]+=.5
    with pytest.raises(ValueError,match='normalization'):m.load_selected_policy(complete)
    packet['normalization']=deepcopy(normalization)
    packet['scope']['train']['subject_ids'].append('p000519')
    with pytest.raises(ValueError,match='checkpoint'):m.load_selected_policy(complete)
