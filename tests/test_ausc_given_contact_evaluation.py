"""Synthetic heldout protocol tests; no TEST asset or native physics is read."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_given_contact_evaluation'
    assert importlib.util.find_spec(name) is not None, 'Once-only paired evaluation is missing'
    return importlib.import_module(name)


@pytest.fixture
def current_native_metadata(tmp_path,monkeypatch):
    """Real producer-shaped JSON/capabilities; no USD, NPZ, actual model or TEST."""
    import json
    from test_ausc_given_contact_lossless_mapping import current_population
    _,rows,original,mapping_ref,processing,report=current_population(tmp_path,monkeypatch)
    m=module();mapping,_=m.deployment._allocation_mapping(mapping_ref,processing)
    chosen=[]
    for split in ('train','validation'):
        for pose in sorted(m.deployment.POSE_TEMPLATES):
            i=next(i for i,row in enumerate(rows) if row['split']==split and row['pose_template']==pose)
            chosen.append(i)
    configs=[];attempts=[]
    for i in chosen:
        config=json.loads(Path(report['plans'][i]['runtime']['path']).read_text())
        config.pop('m2_given_contact_control');config['fresh_expert'].pop('raw_multimodal_capture')
        prep=m.runtime.read_bound(m._file(config['compact_scene_storage']['sources']['preparation_config']))
        config.update({key:prep.get(key) for key in ('geometry_id','geometry_source','geometry_source_sha256','posture_limits','freshly_generated')})
        configs.append(config);attempts.append(deepcopy(rows[i]))
    permit=dict(attempts=attempts,processing=processing,allocation_mapping=mapping_ref)
    return m,configs,permit,mapping,processing,report


def test_current_native_common_cannot_classify_unverified_layered_roles(current_native_metadata):
    m,configs,_,_,_,_=current_native_metadata
    with pytest.raises(ValueError,match='authenticated current RAW'):
        m._native_common(configs[0])


def test_current_authority_refuses_all_four_config_policy_downgrades_before_provenance(current_native_metadata,monkeypatch):
    from sim.ausc_policy import scene_store as store,m2_given_contact_lossless_mapping as lossless
    m,configs,permit,mapping,processing,_=current_native_metadata
    for config in configs:
        config['compact_scene_storage']['dependency_policy']=store.SDK_LAYER_DEPENDENCY_POLICY
        config['compact_scene_storage']['sources'].pop('layered_runtime_equivalence')
    leaves=[];read=lossless._CurrentMetadata.read;plan=m.runtime.read_bound(processing['plan'])
    def track(reader,ref):
        path=lossless._reference(ref)['path']
        if path not in {plan['derivation']['path'],plan['source_recipe']['path']}:leaves.append(path)
        return read(reader,ref)
    monkeypatch.setattr(lossless._CurrentMetadata,'read',track)
    with pytest.raises(ValueError):m._qualified_native_common(configs,permit,mapping,processing)
    assert leaves==[]


def test_genuine_legacy_native_common_keeps_its_exact_existing_projection(tmp_path):
    from test_ausc_given_contact_deployment import scene_metadata_fixture
    _,_,config,mapping,recipe=scene_metadata_fixture(tmp_path)
    config.update(experiment='geometry_qualification',base=str(tmp_path))
    m=module();expected=m._native_common(config)
    recipe_ref=m.runtime.write_new(tmp_path/'legacy-recipe.json',recipe)
    report_ref=m.runtime.write_new(tmp_path/'legacy-derivation.json',dict(schema='m2_given_lossless_derivation_map_v1',
        authoring_request=dict(selected=[]),scenes=[]))
    plan=m.runtime.write_new(tmp_path/'legacy-plan.json',dict(schema=m.runtime.dataset.PLAN_SCHEMA,
        source_recipe=recipe_ref,derivation=report_ref,accounting_scope=dict(extra_input_refs=[
            dict(category='config',reference=ref) for ref in (recipe_ref,report_ref)])))
    processing=dict(plan=plan)
    assert m._qualified_native_common([deepcopy(config) for _ in range(4)],dict(processing=processing),mapping,processing)==expected
    assert 'heldout_provenance' not in expected


def test_current_family_missing_false_typed_and_contradictory_authority_cannot_select_legacy(current_native_metadata,monkeypatch):
    import json
    from sim.ausc_policy import scene_store as store,m2_given_contact_lossless_mapping as lossless
    from test_ausc_given_contact_lossless_mapping import rebind_current_json
    m,configs,original_permit,mapping,processing,_=current_native_metadata
    current_configs=deepcopy(configs)
    for config in configs:
        config['compact_scene_storage']['dependency_policy']=store.SDK_LAYER_DEPENDENCY_POLICY
        config['compact_scene_storage']['sources'].pop('layered_runtime_equivalence')
    plan_path=Path(processing['plan']['path']);plan_raw=plan_path.read_bytes();base=json.loads(plan_raw)
    recipe_path=Path(base['source_recipe']['path']);recipe_raw=recipe_path.read_bytes()
    report_path=Path(base['derivation']['path']);report_raw=report_path.read_bytes()
    top={str(recipe_path),str(report_path)};leaves=[];read=lossless._CurrentMetadata.read
    def track(reader,ref):
        path=lossless._reference(ref)['path']
        if path not in top:leaves.append(path)
        return read(reader,ref)
    monkeypatch.setattr(lossless._CurrentMetadata,'read',track)
    faults=('plan_missing','plan_false','plan_int','recipe_missing','recipe_false','recipe_int',
        'recipe_policy','request_false','request_int','cpu_false','cpu_policy','records_missing','mapping_only')
    try:
        for fault in faults:
            plan=json.loads(plan_raw);recipe=json.loads(recipe_raw);report=json.loads(report_raw)
            if fault=='plan_missing':plan.pop('current_raw_equivalence')
            elif fault=='plan_false':plan['current_raw_equivalence']=False
            elif fault=='plan_int':plan['current_raw_equivalence']=1
            elif fault=='recipe_missing':recipe.pop('current_raw_equivalence')
            elif fault=='recipe_false':recipe['current_raw_equivalence']=False
            elif fault=='recipe_int':recipe['current_raw_equivalence']=1
            elif fault=='recipe_policy':recipe['scene_dependency_policy']=store.SDK_LAYER_DEPENDENCY_POLICY
            elif fault=='request_false':report['authoring_request']['current_raw_equivalence']=False
            elif fault=='request_int':report['authoring_request']['current_raw_equivalence']=1
            elif fault=='cpu_false':report['cpu_execution']['current_raw_equivalence']=False
            elif fault=='cpu_policy':report['cpu_execution']['dependency_policy']=store.SDK_LAYER_DEPENDENCY_POLICY
            elif fault=='records_missing':report.pop('current_raw_prepared')
            else:
                plan.pop('current_raw_equivalence');recipe={};report={}
            rebind_current_json(plan,plan['source_recipe'],recipe);rebind_current_json(plan,plan['derivation'],report)
            plan_path.write_text(json.dumps(plan));authority=deepcopy(processing);authority['plan']=m.runtime.reference(plan_path)
            permit=deepcopy(original_permit);permit['processing']=authority
            for candidates in (configs,current_configs):
                with pytest.raises(ValueError):m._qualified_native_common(candidates,permit,mapping,authority)
            assert leaves==[],fault
    finally:
        plan_path.write_bytes(plan_raw);recipe_path.write_bytes(recipe_raw);report_path.write_bytes(report_raw)


def test_four_current_scenes_project_only_authenticated_shared_conditions(current_native_metadata,monkeypatch):
    from sim.ausc_policy import scene_store as store
    m,configs,permit,mapping,processing,_=current_native_metadata
    read=Path.read_bytes
    def only_json(path):
        assert path.suffix=='.json' and path.name!='collision_geometry.json', 'Projection opened a payload'
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',only_json)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Projection decoded body'))
    common=m._qualified_native_common(configs,permit,mapping,processing)
    assert common['schema']==store.SDK_SCHEMA and common['dependency_policy']==store.SDK_DEPENDENCY_POLICY
    evidence=common['heldout_provenance']
    assert evidence['schema']=='m2_given_contact_current_raw_to_dense_sdk_native_common_v1'
    assert len(evidence['qualified_scenes'])==4
    assert len({case['layered_runtime_equivalence']['file'] for case in evidence['qualified_scenes']})==4
    assert len({(case['split'],case['subject_id']) for case in evidence['qualified_scenes']})==4
    assert not {'scene','scene_base','scene_authoring_receipt','layered_runtime_equivalence'}&set(common['shared_sources'])
    assert common['physics']==configs[0]['physics']
    assert common['fresh']=={k:v for k,v in configs[0]['fresh_expert'].items() if k!='plans'}


def test_current_projection_rejects_last_header_before_any_provenance_leaf(current_native_metadata,monkeypatch):
    from sim.ausc_policy import m2_given_contact_lossless_mapping as lossless
    m,original,permit,mapping,processing,_=current_native_metadata
    for fault in ('policy','missing','unknown','unregistered','typed_bytes','sdk','camera','physical','fresh'):
        configs=deepcopy(original);sources=configs[-1]['compact_scene_storage']['sources']
        if fault=='policy':configs[-1]['compact_scene_storage']['dependency_policy']='hash_bound_static_usd_and_isaacsim60_mdl_v3'
        elif fault=='missing':sources.pop('layered_runtime_equivalence')
        elif fault=='unknown':sources['unknown_scene_provenance']=deepcopy(sources['scene_base'])
        elif fault=='unregistered':sources['layered_runtime_equivalence']['file']='/unregistered/sidecar.json'
        elif fault=='typed_bytes':sources['scene_base']['bytes']=float(sources['scene_base']['bytes'])
        elif fault=='sdk':sources['sdk_mdl_0000']['sha256']='f'*64
        elif fault=='camera':sources['approved_camera']['sha256']='f'*64
        elif fault=='physical':configs[-1]['physics']['position_iterations']=32
        else:configs[-1]['fresh_expert']['collision_profile']='none'
        leaves=[];read=lossless._CurrentMetadata.read
        def record(reader,ref):
            path=lossless._reference(ref)['path']
            plan=m.runtime.read_bound(processing['plan'])
            if path not in {plan['derivation']['path'],plan['source_recipe']['path']}:leaves.append(path)
            return read(reader,ref)
        with monkeypatch.context() as patch:
            patch.setattr(lossless._CurrentMetadata,'read',record)
            with pytest.raises(ValueError):m._qualified_native_common(configs,permit,mapping,processing)
        assert leaves==[], fault


def test_dense_heldout_rebuilds_every_scene_role_and_never_inherits_layered_provenance(current_native_metadata):
    from sim.ausc_policy import scene_store as store
    m,configs,permit,mapping,processing,_=current_native_metadata
    common=m._qualified_native_common(configs,permit,mapping,processing)
    root=Path(processing['plan']['path']).parent/'synthetic-heldout-prepared'
    ref=lambda name:dict(path=str(root/name),bytes=1,sha256='e'*64)
    files={filename:ref(filename) for filename in store.FILES.values()}
    row=dict(preparation_config=ref('frozen-prepare.json'),prepared_complete=ref('COMPLETE.json'),inventory=dict(files=files))
    binding=m._binding_from_inventory(dict(native_common=common),row,'synthetic-heldout')
    assert (binding['schema'],binding['dependency_policy'])==(store.SDK_SCHEMA,store.SDK_DEPENDENCY_POLICY)
    assert binding['sources']['scene']['file']==str(root/store.FILES['scene'])
    for role,name in store.FILES.items():assert binding['sources'][role]['file']==str(root/name)
    assert not {'scene_base','scene_authoring_receipt','layered_runtime_equivalence'}&set(binding['sources'])
    assert set(binding['sources'])==set(common['shared_sources'])|set(store.FILES)|{'preparation_config','prepared_complete'}
    changed=deepcopy(common);changed['shared_sources']['scene_base']=deepcopy(configs[0]['compact_scene_storage']['sources']['scene_base'])
    with pytest.raises(ValueError):m._binding_from_inventory(dict(native_common=changed),row,'synthetic-heldout')


def test_current_native_common_declaration_refuses_fake_missing_mixed_and_aliased_provenance(current_native_metadata):
    m,configs,permit,mapping,processing,_=current_native_metadata
    common=m._qualified_native_common(configs,permit,mapping,processing)
    for fault in ('empty_case','unknown_case','alias_case','index_type','sidecar_policy','source_missing','source_index','unknown_record'):
        changed=deepcopy(common);record=changed['heldout_provenance'];case=record['qualified_scenes'][-1]
        if fault=='empty_case':record['qualified_scenes'][-1]={}
        elif fault=='unknown_case':case['unknown']=True
        elif fault=='alias_case':record['qualified_scenes'][-1]=deepcopy(record['qualified_scenes'][0])
        elif fault=='index_type':case['scene_index']=float(case['scene_index'])
        elif fault=='sidecar_policy':case['sidecar']['dependency_policy']='hash_bound_static_usd_and_isaacsim60_mdl_v3'
        elif fault=='source_missing':case['scene_sources'].pop('scene_base')
        elif fault=='source_index':case['sidecar']['scene_index']+=1
        else:record['unknown']=False
        with pytest.raises(ValueError):m._dense_current_common(changed)


def test_metadata_allocation_covers_every_test_identity_once_and_balances_known_poses():
    m=module();ids=['p%06d'%i for i in range(34)]
    counts={target:i for i,target in enumerate(m.TARGETS)}
    rows=m.allocate_bodies(ids,seed=20261029,train_feasibility=counts)
    assert rows==m.allocate_bodies(list(reversed(ids)),seed=20261029,train_feasibility=counts)
    assert [r['subject_id'] for r in rows]==sorted(ids)
    assert len({r['seed'] for r in rows})==34
    assert [r['split'] for r in rows]==['test']*34
    assert {pose:sum(r['pose_template']==pose for r in rows) for pose in m.deployment.POSE_TEMPLATES}==dict.fromkeys(m.deployment.POSE_TEMPLATES,17)
    assert sum(r['arm_order']==['expert','act'] for r in rows)==17
    assert sum(r['arm_order']==['act','expert'] for r in rows)==17
    assert all(set(r['target_priority'])==set(m.TARGETS) and len(r['target_priority'])==10 for r in rows)
    heads={t:sum(r['target_priority'][0]==t for r in rows) for t in m.TARGETS}
    assert set(heads.values())<={3,4}


@pytest.mark.parametrize('fault',['missing_identity','duplicate_identity','negative_seed','bool_seed','missing_target','noninteger_count','negative_count'])
def test_metadata_allocation_refuses_incomplete_or_outcome_like_inputs(fault):
    m=module();ids=['p%06d'%i for i in range(34)];seed=20261029;counts=dict.fromkeys(m.TARGETS,1)
    if fault=='missing_identity':ids.pop()
    elif fault=='duplicate_identity':ids[1]=ids[0]
    elif fault=='negative_seed':seed=-1
    elif fault=='bool_seed':seed=True
    elif fault=='missing_target':counts.pop(m.TARGETS[0])
    elif fault=='noninteger_count':counts[m.TARGETS[0]]=.95
    else:counts[m.TARGETS[0]]=-1
    with pytest.raises(ValueError):m.allocate_bodies(ids,seed=seed,train_feasibility=counts)


def test_first_feasible_selection_retains_every_target_and_does_not_use_native_results():
    m=module();priority=list(reversed(m.TARGETS))
    outcomes=[dict(target_id=t,status='found' if t in priority[3:5] else 'bounded_planning_unfound') for t in m.TARGETS]
    assert m.select_first_feasible(priority,outcomes)==priority[3]
    outcomes=[dict(target_id=t,status='bounded_planning_unfound') for t in m.TARGETS]
    assert m.select_first_feasible(priority,outcomes) is None


@pytest.mark.parametrize('fault',['missing','duplicate','unknown','native_status','priority','outcome_field'])
def test_first_feasible_selection_refuses_screening_or_hidden_denominator(fault):
    m=module();priority=list(m.TARGETS)
    outcomes=[dict(target_id=t,status='found') for t in m.TARGETS]
    if fault=='missing':outcomes.pop()
    elif fault=='duplicate':outcomes[-1]=outcomes[0]
    elif fault=='unknown':outcomes[0]['target_id']='L6'
    elif fault=='native_status':outcomes[0]['status']='native_pass'
    elif fault=='priority':priority[-1]=priority[0]
    else:outcomes[0]['native_success']=True
    with pytest.raises(ValueError):m.select_first_feasible(priority,outcomes)


def test_exact_clopper_pearson_exposes_zero_and_small_denominators():
    m=module()
    assert m.binary_rate(0,0)==dict(successes=0,denominator=0,rate=None,confidence=.95,interval=None)
    zero=m.binary_rate(0,34);full=m.binary_rate(34,34)
    assert zero['interval'][0]==0 and full['interval'][1]==1
    assert zero['interval'][1]==pytest.approx(1-.025**(1/34))
    assert full['interval'][0]==pytest.approx(.025**(1/34))
    with pytest.raises(ValueError):m.binary_rate(35,34)


def test_fixed_video_selection_is_in_original_body_order_and_includes_failures():
    m=module();rows=[dict(subject_id='p%06d'%i,arm=arm,launched=True,success=i%2==0,
        video=dict(path='/synthetic/%s-%s.mp4'%(i,arm),sha256='a'*64)) for i in range(34) for arm in ('expert','act')]
    selected=m.select_review_videos(rows,body_ids=['p%06d'%i for i in range(34)])
    assert [(r['subject_id'],r['arm']) for r in selected]==[(r['subject_id'],r['arm']) for r in rows[:4]]
    assert selected[2]['success'] is False
    assert selected==m.select_review_videos(rows,body_ids=['p%06d'%i for i in range(34)])


def native_tick_fixture(tmp_path):
    m=module();state=dict(time_s=0.,q=np.zeros(7),tcp_base=np.array([0.,0.,.1]),rotation_world=np.eye(3))
    env=SimpleNamespace(rows=[],base=np.eye(4),state=lambda:deepcopy(state))
    controller=SimpleNamespace(env=env,trajectory=None,control_owner='PREPARATION',actor_phase=None,
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='FREE'))),
        metadata=dict(reference_sha256='a'*64),_current_intent_audit=None)
    audit=m.NativeAudit(tmp_path/'native.jsonl',controller.metadata)
    return m,audit,controller,state


def test_neutral_native_hook_records_actual_same_time_full_pose_and_calls_original_tick_once(tmp_path):
    m,audit,c,state=native_tick_fixture(tmp_path);calls=[]
    def tick():
        calls.append(1);state['time_s']=1/240.;state['q'][2]=.01
        state['rotation_world']=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        c.env.rows.append(dict(time_s=state['time_s'],q=state['q'].tolist(),q_command=[0.]*7,
            phase='FREE',subphase='staging',physical_violations=[]))
        return 'original result'
    assert audit.tick(c,[0.]*7,'staging',tick)=='original result'
    receipt=audit.close();row=m.runtime.read_bound(receipt['index'])
    assert len(calls)==1 and receipt['rows']==1 and row['rows']==1
    import json
    event=json.loads((tmp_path/'native.jsonl').read_text())
    assert event['observed_before']['physics_row']==0 and event['observed_after']['physics_row']==1
    assert event['observed_before']['time_s']==0 and event['observed_after']['time_s']==1/240.
    assert event['observed_after']['pose_base'][0][1]==-1
    assert event['observed_after']['q'][2]==.01 and event['q_command']==[0.]*7
    assert not hasattr(c.env,'raw_capture')


@pytest.mark.parametrize('fault',['two_steps','stale_state','q_readback','wrong_command'])
def test_neutral_hook_refuses_invented_or_stale_after_step_evidence(tmp_path,fault):
    m,audit,c,state=native_tick_fixture(tmp_path)
    def tick():
        state['time_s']=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=[0.]*7,q_command=[0.]*7,phase='FREE',subphase='staging',physical_violations=[]))
        if fault=='two_steps':c.env.rows.append(deepcopy(c.env.rows[-1]))
        elif fault=='stale_state':state['time_s']=0.
        elif fault=='q_readback':c.env.rows[-1]['q'][0]=.01
        else:c.env.rows[-1]['q_command'][0]=.01
    with pytest.raises(ValueError):audit.tick(c,[0.]*7,'staging',tick)
    audit.close()


def test_neutral_hook_preserves_executed_error_tick_and_original_error_identity(tmp_path):
    m,audit,c,state=native_tick_fixture(tmp_path)
    class OriginalSafetyError(RuntimeError):pass
    def tick():
        state['time_s']=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=[0.]*7,q_command=[0.]*7,phase='FREE',subphase='staging',physical_violations=['original gate']))
        raise OriginalSafetyError('original gate')
    with pytest.raises(OriginalSafetyError,match='original gate'):audit.tick(c,[0.]*7,'staging',tick)
    receipt=audit.close()
    assert receipt['rows']==1 and receipt['errors'] and receipt['complete'] is False
    partial=m._read_native_events(receipt,c.metadata)
    assert len(partial)==1 and partial[0]['physical_violations']==['original gate']
    with pytest.raises(ValueError):m.read_native_trace(receipt,c.metadata)


def test_neutral_hook_records_no_physics_for_a_precommand_rejection(tmp_path):
    m,audit,c,state=native_tick_fixture(tmp_path)
    def tick():raise RuntimeError('command rejected')
    with pytest.raises(RuntimeError,match='command rejected'):audit.tick(c,[0.]*7,'staging',tick)
    receipt=audit.close()
    assert receipt['rows']==0 and receipt['complete'] is False and receipt['errors']


def stop_profile_fixture(tmp_path,monkeypatch):
    """Exact v3 input API contract; this fixture never claims native Ruckig."""
    from test_ausc_given_contact_handoff import FixtureTrajectory
    m,audit,c,state=native_tick_fixture(tmp_path);h=m.runtime.handoff
    stop=dict(h.PLANNING_LIMITS,jerk=1.49)
    monkeypatch.setattr(h,'STOP_PLANNING_LIMITS',stop,raising=False)
    monkeypatch.setattr(h,'POLICY_ID','m2_given_interior_parking_handoff_v3')
    def snapshot(t,*,profile='l2'):
        limits=h.PLANNING_LIMITS if profile=='l2' else stop
        m.need(profile in ('l2','stop') and t.input.max_jerk==[limits['jerk']]*7 and
            t.input.max_velocity==[limits['velocity']]*7 and t.input.max_acceleration==[limits['acceleration']]*7,
            'Frozen exact v3 planning profile differs')
        return dict(dt=t.dt,fields={k:h._encode(getattr(t.input,k)) for k in h.INPUT_FIELDS},
            velocity=t.velocity.tolist(),acceleration=t.acceleration.tolist(),ruckig_version=t.module.__version__)
    monkeypatch.setattr(h,'input_snapshot',snapshot)
    c.trajectory=FixtureTrajectory(np.zeros(7),dt=1/240.,**h.PLANNING_LIMITS)
    before=snapshot(c.trajectory);c.trajectory.input.max_jerk=[1.49]*7;after=snapshot(c.trajectory,profile='stop')
    transition=dict(schema='m2_given_l2_to_stop_profile_transition_v3',handoff_policy_id=h.POLICY_ID,
        from_profile=deepcopy(h.PLANNING_LIMITS),to_profile=stop,changed_fields=['max_jerk'],
        before=before,before_sha256=h.digest(before),after=after,after_sha256=h.digest(after))
    c.l2_handoffs=[dict(parking_stop_profile_transition=transition)]
    return m,audit,c,state,transition


@pytest.mark.parametrize('owner',['L2_STOP','L3_ALIGN','CONTACT','RETURN'])
def test_neutral_observer_preserves_explicit_stop_profile_after_handoff(tmp_path,monkeypatch,owner):
    m,audit,c,state,transition=stop_profile_fixture(tmp_path,monkeypatch);c.control_owner=owner
    original=m._interpolator(c)
    assert original['planning_profile']=='stop'
    assert original['stop_profile_transition_sha256']==m.runtime.handoff.digest(transition)
    def tick():
        state['time_s']=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=[0.]*7,q_command=[0.]*7,phase='FREE',subphase='l3_align',physical_violations=[]))
    audit.tick(c,[0.]*7,'l3_align',tick)
    assert m._interpolator(c)==original and audit.close()['rows']==1


def test_neutral_observer_keeps_original_l2_profile_without_a_stop_transition(tmp_path,monkeypatch):
    m,audit,c,_,_=stop_profile_fixture(tmp_path,monkeypatch)
    c.trajectory.input.max_jerk=[1.5]*7;c.control_owner='ACT_L2';c.l2_handoffs=[]
    snapshot=m._interpolator(c)
    assert snapshot['planning_profile']=='l2' and snapshot['stop_profile_transition_sha256'] is None
    assert c.trajectory.input.max_jerk==[1.5]*7
    audit.close()


@pytest.mark.parametrize('fault',['missing_transition','changed_field','changed_hash','unknown_bound','ordinary_l2'])
def test_neutral_observer_never_falls_back_between_l2_and_stop_profiles(tmp_path,monkeypatch,fault):
    m,audit,c,_,transition=stop_profile_fixture(tmp_path,monkeypatch);c.control_owner='L3_ALIGN'
    if fault=='missing_transition':c.l2_handoffs=[]
    elif fault=='changed_field':
        transition['after']['fields']['target_position'][0]=.01
        transition['after_sha256']=m.runtime.handoff.digest(transition['after'])
    elif fault=='changed_hash':transition['after_sha256']='a'*64
    elif fault=='unknown_bound':c.trajectory.input.max_jerk=[1.48]*7
    else:c.control_owner='ACT_L2'
    with pytest.raises(ValueError):m._interpolator(c)
    audit.close()


def metadata_layout():
    m=module();rows=m.allocate_bodies(['p%06d'%i for i in range(34)],seed=20261029,
        train_feasibility=dict.fromkeys(m.TARGETS,1))
    return m,m.preparation_layout(rows,root='/synthetic/heldout-metadata-v1')


def planning_records(layout,found=32):
    result=[]
    for index,row in enumerate(layout['bodies']):
        for target in row['target_priority']:
            hit=index<found and target in row['target_priority'][2:4]
            result.append(dict(subject_id=row['subject_id'],target_id=target,
                status='found' if hit else 'bounded_planning_unfound',
                planner_result=dict(path=row['planning'][target]['complete'],bytes=123,sha256='a'*64),
                fresh_plan=(dict(path=row['planning'][target]['fresh_plan'],bytes=456,sha256='b'*64) if hit else None)))
    return result


def test_preparation_paths_cover_all34_and340_without_reading_any_payload(monkeypatch):
    m=module()
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('metadata helper read a payload'))
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('metadata helper read a payload'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata helper decoded a payload'))
    _,layout=metadata_layout()
    assert layout['schema']==m.PREPARATION_SCHEMA and len(layout['bodies'])==34
    assert layout['test_payload_access_authorized'] is layout['native_access_authorized'] is False
    assert layout['metadata_only'] is True
    paths=[]
    for row in layout['bodies']:
        assert row['split']=='test' and set(row['planning'])==set(m.TARGETS)
        paths.extend([row['preparation_config'],row['prepared_output']])
        for target in row['target_priority']:
            paths.extend(row['planning'][target].values())
    assert len(paths)==len(set(paths))
    records=planning_records(layout)
    inventory=m.planning_inventory(layout,records)
    assert len(inventory['outcomes'])==340 and inventory['available_bodies']==32
    assert inventory['payloads_verified'] is inventory['native_access_authorized'] is False


@pytest.mark.parametrize('fault',['missing_body','duplicate_body','pose_balance','arm_balance','priority','seed','relative_root'])
def test_preparation_layout_rejects_incomplete_or_changed_body_plan(fault):
    m=module();rows=m.allocate_bodies(['p%06d'%i for i in range(34)],seed=20261029,
        train_feasibility=dict.fromkeys(m.TARGETS,1));root='/synthetic/future'
    if fault=='missing_body':rows.pop()
    elif fault=='duplicate_body':rows[-1]=rows[0]
    elif fault=='pose_balance':rows[0]['pose_template']='unseen-pose'
    elif fault=='arm_balance':rows[0]['arm_order']=['expert','expert']
    elif fault=='priority':rows[0]['target_priority'][0]=rows[0]['target_priority'][1]
    elif fault=='seed':rows[0]['seed']=True
    else:root='../future'
    with pytest.raises(ValueError):m.preparation_layout(rows,root=root)


@pytest.mark.parametrize('kind',['directory','file','broken_link'])
def test_future_outputs_must_be_absent_without_opening_them(tmp_path,kind):
    m,layout=metadata_layout();layout=m.preparation_layout([
        {k:v for k,v in row.items() if k not in ('preparation_config','prepared_output','planning')}
        for row in layout['bodies']],root=str(tmp_path/'metadata'))
    path=Path(layout['bodies'][0]['prepared_output']);path.parent.mkdir(parents=True)
    if kind=='directory':path.mkdir()
    elif kind=='file':path.write_text('existing must not be reused')
    else:path.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError,match='already exists'):m.require_future_outputs_absent(layout)


@pytest.mark.parametrize('fault',['missing','duplicate','reordered','native_status','extra_outcome','alias_report','alias_plan','unfound_plan'])
def test_all340_planning_slots_and_exact_future_roles_are_required(fault):
    m,layout=metadata_layout();records=planning_records(layout)
    if fault=='missing':records.pop()
    elif fault=='duplicate':records[-1]=records[0]
    elif fault=='reordered':records[0],records[1]=records[1],records[0]
    elif fault=='native_status':records[0]['status']='native_pass'
    elif fault=='extra_outcome':records[0]['physical_success']=True
    elif fault=='alias_report':records[0]['planner_result']['path']='/synthetic/body.npz'
    elif fault=='alias_plan':records[2]['fresh_plan']['path']='/synthetic/alias.json'
    else:records[0]['fresh_plan']=dict(path='/synthetic/unused.json',bytes=1,sha256='c'*64)
    with pytest.raises(ValueError):m.planning_inventory(layout,records)


def test_fixed_pair_schedule_keeps_absent_bodies_and_both_original_arms():
    m,layout=metadata_layout();inventory=m.planning_inventory(layout,planning_records(layout,found=32))
    value=m.paired_schedule(layout,inventory,owner_run_id='heldout-pairs-v1')
    assert value['body_denominator']==34 and len(value['pairs'])==34 and len(value['attempts'])==64
    assert value['native_access_authorized'] is False and value['payloads_verified'] is False
    assert sum(pair['selected_target'] is None for pair in value['pairs'])==2
    for row,pair in zip(layout['bodies'],value['pairs']):
        if pair['selected_target'] is None:
            assert pair['attempts']==[]
        else:
            assert pair['selected_target']==row['target_priority'][2]
            assert [r['arm'] for r in pair['attempts']]==row['arm_order']
            assert len({r['fresh_plan']['sha256'] for r in pair['attempts']})==1
            assert all(r['seed']==row['seed'] and r['pose_template']==row['pose_template'] for r in pair['attempts'])
    assert len(m.paired_schedule(layout,m.planning_inventory(layout,planning_records(layout,34)),
        owner_run_id='heldout-pairs-v1')['attempts'])==68


def test_changed_inventory_cannot_reselect_targets_or_turn_metadata_into_capability():
    m,layout=metadata_layout();value=m.planning_inventory(layout,planning_records(layout))
    value['selected'][0]['target_id']=layout['bodies'][0]['target_priority'][3]
    with pytest.raises(ValueError):m.paired_schedule(layout,value,owner_run_id='heldout-pairs-v1')
    value=m.planning_inventory(layout,planning_records(layout));value['native_access_authorized']=True
    with pytest.raises(ValueError):m.paired_schedule(layout,value,owner_run_id='heldout-pairs-v1')


def test_fixed_video_selection_never_replaces_absent_first_body_with_later_success():
    m=module();ids=['p%06d'%i for i in range(34)]
    records=[dict(subject_id=ids[1],arm='act',video=None),
             dict(subject_id=ids[2],arm='expert',video={'path':'/synthetic/best.mp4'})]
    selected=m.select_review_videos(records,body_ids=ids)
    assert [(r['subject_id'],r['arm']) for r in selected]==[(i,a) for i in ids[:2] for a in ('expert','act')]
    assert all(row.get('video') is None for row in selected)
    assert selected[0]['availability']=='native_arm_absent_or_unreported'


def reported_outcomes(schedule):
    return [dict(subject_id=a['subject_id'],arm=a['arm'],job_id=a['job_id'],launched=True,
        physics_steps=100,status='passed') for a in schedule['attempts']]


@pytest.mark.parametrize('executed_pairs',[29,30,34])
def test_pair_milestone_preserves_full34_and_actual_two_arm_execution_denominators(executed_pairs):
    m,layout=metadata_layout();inv=m.planning_inventory(layout,planning_records(layout,34))
    schedule=m.paired_schedule(layout,inv,owner_run_id='fixed-heldout-owner')
    outcomes=reported_outcomes(schedule)
    for row in outcomes[executed_pairs*2:]:
        row.update(launched=False,physics_steps=0,status='planned_unlaunched')
    result=m.reported_pair_counts(schedule,outcomes)
    assert result['body_denominator']==34 and result['available_bodies']==34
    assert result['executed_pairs']==executed_pairs
    assert result['reported_milestone_reached'] is (executed_pairs>=30)
    assert result['expert_success']['denominator']==result['act_success']['denominator']==executed_pairs
    assert result['paired']==dict(both=executed_pairs,only_expert=0,only_act=0,neither=0)
    assert result['native_evidence_verified'] is result['heldout_evaluation_complete'] is False


def test_failed_aborted_and_unlaunched_mates_are_retained_in_counts():
    m,layout=metadata_layout();inv=m.planning_inventory(layout,planning_records(layout,32))
    schedule=m.paired_schedule(layout,inv,owner_run_id='fixed-heldout-owner');rows=reported_outcomes(schedule)
    first={r['arm']:r for r in rows[:2]};first['act']['status']='physical_failure'
    second={r['arm']:r for r in rows[2:4]};second['expert'].update(status='infrastructure_abort',physics_steps=0)
    third={r['arm']:r for r in rows[4:6]};third['act'].update(status='missing_result',physics_steps=None)
    result=m.reported_pair_counts(schedule,rows)
    assert result['body_denominator']==34 and result['available_bodies']==32 and result['executed_pairs']==30
    assert result['paired']==dict(both=29,only_expert=1,only_act=0,neither=0)
    assert result['status_counts']['physical_failure']==result['status_counts']['infrastructure_abort']==result['status_counts']['missing_result']==1
    assert result['availability']['denominator']==34 and result['availability']['successes']==32
    assert result['act_success']['successes']==29 and result['act_success']['denominator']==30


@pytest.mark.parametrize('fault',['missing','duplicate','reordered','unknown_status','success_without_launch','success_without_physics','invented_steps','changed_arm'])
def test_reported_pair_counts_refuse_hidden_or_fabricated_denominators(fault):
    m,layout=metadata_layout();schedule=m.paired_schedule(layout,m.planning_inventory(layout,planning_records(layout,34)),owner_run_id='fixed-owner')
    rows=reported_outcomes(schedule)
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=rows[0]
    elif fault=='reordered':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='unknown_status':rows[0]['status']='good_enough'
    elif fault=='success_without_launch':rows[0]['launched']=False
    elif fault=='success_without_physics':rows[0]['physics_steps']=0
    elif fault=='invented_steps':rows[0]['physics_steps']=True
    else:rows[0]['arm']='teacher_fallback'
    with pytest.raises(ValueError):m.reported_pair_counts(schedule,rows)


def test_config_projection_excludes_only_declared_capability_references():
    m=module();ref=dict(path='/synthetic/native-capability.json',bytes=1,sha256='a'*64)
    cfg=dict(physics={'original':1},patient_asset={'expected_split':'test'},
        m2_given_contact_evaluation=dict(schema=m.NATIVE_ROUTE_SCHEMA,capability=ref,episode_id='test-episode',arm='act'),
        m2_given_contact_deployment=dict(permit=ref,metadata={'full_R':'unchanged'},simulation_only=True))
    original=deepcopy(cfg);projected=m.configuration_projection(cfg)
    assert projected['physics']==original['physics'] and projected['patient_asset']==original['patient_asset']
    assert 'capability' not in projected[m.ROUTE_KEY] and 'permit' not in projected['m2_given_contact_deployment']
    assert projected['m2_given_contact_deployment']['metadata']==original['m2_given_contact_deployment']['metadata']
    assert cfg==original
    cfg['physics']['original']=2
    assert m.runtime.digest(m.configuration_projection(cfg))!=m.runtime.digest(projected)
    cfg['m2_given_contact_deployment']['permit']=dict(ref,sha256='b'*64)
    with pytest.raises(ValueError):m.configuration_projection(cfg)


@pytest.mark.parametrize('stage',['preparation','native'])
def test_unissued_test_capability_refuses_before_payload_read(stage,monkeypatch):
    m=module()
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unauthorized TEST payload decoded'))
    cfg={m.ROUTE_KEY:dict(schema=m.PREP_ROUTE_SCHEMA if stage=='preparation' else m.NATIVE_ROUTE_SCHEMA,
        capability={'path':'/synthetic/unissued.json','sha256':'0'*64},subject_id='p000000')}
    with pytest.raises((ValueError,FileNotFoundError)):m.preflight(cfg)


@pytest.fixture
def sealed_metadata(tmp_path,monkeypatch):
    """Mock only the actual-model/smoke authority. No real TEST identities/assets."""
    m=module();ids=['p%06d'%i for i in range(34)]
    admission=m.runtime.write_new(tmp_path/'admission.json',{'synthetic_fixture_only':True})
    smoke=m.runtime.write_new(tmp_path/'smokes.json',{'synthetic_fixture_only':True})
    templates={pose:dict(prepare_only=True,physics={'unchanged':1},fresh_expert={},
        patient_asset={'expected_split':'train'}) for pose in m.deployment.POSE_TEMPLATES}
    for template in templates.values():
        template.update(geometry_id=m.runtime.geometry.GEOMETRY_ID,
            geometry_source_sha256=m.runtime.geometry.GEOMETRY_SOURCE_SHA256,
            geometry_source='/synthetic/original.usdc',freshly_generated=True,posture_limits={'unchanged':True})
        template['patient_asset']['identity_split']=str(m.runtime.shared.contract.IDENTITY_SPLIT_PATH)
    authority=dict(admission=admission,model={'fixture':'not a qualified model'},processing={},source_inventory={},
        train_counts=dict.fromkeys(m.TARGETS,1),templates=templates,pose_sources={pose:pose for pose in m.deployment.POSE_TEMPLATES},native_common=dict(schema='m2_compact_scene_storage_v1',
            store_root=str(tmp_path/'cas'),dependency_policy='builtin_omnipbr_only_v1',shared_sources={},
            fresh=dict(collision_profile='full',native_retainer_audit={'synthetic_only':True}),
            experiment='geometry_qualification',base=str(tmp_path),isaac_python='/synthetic/bin/python'))
    for template in templates.values():
        template.update(base=str(tmp_path),isaac_python='/synthetic/bin/python',asset_root='/synthetic/assets',
            ld_library_path='/synthetic/lib',time_limit_seconds=1800)
    monkeypatch.setattr(m,'_verified_prerequisites',lambda a,s:deepcopy(authority))
    monkeypatch.setattr(m,'_identity_metadata',lambda:ids)
    specs={}
    allocation=m.allocate_bodies(ids,seed=20261029,train_feasibility=authority['train_counts'])
    for row in allocation:
        directory=tmp_path/row['subject_id'];directory.mkdir()
        manifest=m.runtime.write_new(directory/'asset_manifest.json',[dict(file='body.npz',identity=row['subject_id'],
            split='test',sha256='a'*64,numeric_candidate_screen_passed=True,mesh_scale=1.,source_pose=row['pose_template'])])
        summary=m.runtime.write_new(directory/'summary.json',dict(schema='m2_observed_seated_retarget_v1',
            identity_split_sha256=m.runtime.shared.contract.IDENTITY_SPLIT_SHA256))
        specs[row['subject_id']]=dict(expected_split='test',directory=str(directory),file='body.npz',
            identity_split=str(m.runtime.shared.contract.IDENTITY_SPLIT_PATH),manifest_sha256=manifest['sha256'],summary_sha256=summary['sha256'])
    root=str(tmp_path/'planned')
    body=m.freeze_body_plan(admission,smoke,seed=20261029,patient_specs=specs,root=root,output=tmp_path/'body-plan.json')
    plan=m.runtime.read_bound(body);configs=[]
    for row in plan['layout']['bodies']:
        config=deepcopy(templates[row['pose_template']]);config['patient_asset']=specs[row['subject_id']]
        config[m.ROUTE_KEY]=dict(schema=m.PREP_ROUTE_SCHEMA,capability=None,subject_id=row['subject_id'],
            pose_template=row['pose_template'],seed=row['seed'])
        configs.append(config)
    return m,body,configs,tmp_path


def test_preparation_freeze_uses_projection_and_never_hashes_nonexistent_outputs(sealed_metadata,monkeypatch):
    m,body,configs,tmp_path=sealed_metadata
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata freeze decoded body'))
    plan=m.freeze_preparation_plan(body,configs,output=tmp_path/'preparation-plan.json')
    inventory=m.runtime.read_bound(plan)
    assert len(inventory['configurations'])==34
    assert all(not Path(row['output']).exists() for row in inventory['configurations'])
    issued=m.issue_preparation_capability(plan,output=tmp_path/'planned'/'preparation-capability.json')
    aggregate=m.runtime.read_bound(issued['owner_configuration']);owner=m.runtime.read_bound(body)['layout']['preparation_owner']
    assert aggregate[m.AGGREGATE_KEY]['stage']=='preparation' and issued['owner_configuration']['path']==owner['config']
    assert m.planning_aggregate_preflight(aggregate,output=owner['review'])['stage']=='preparation'
    assert all(Path(row['output']).parent==Path(owner['review'])/'prepared' for row in issued['configurations'])
    for row in issued['configurations']:
        config=m.runtime.read_bound(row['config'])
        assert config[m.ROUTE_KEY]['capability']==issued['capability']
        assert m.preflight(config,output=row['output'])['row']['subject_id']==row['subject_id']
    with pytest.raises(ValueError):m.verify_capability(issued['capability'],stage='native')


def test_body_plan_authenticates_structured_known_train_pose_and_fixed_patient_roles_before_payload(sealed_metadata,monkeypatch):
    m,ref,configs,tmp=sealed_metadata;old=m.runtime.read_bound(ref)
    authority=m._verified_prerequisites(old['admission'],old['bridge_smokes'])
    authority['pose_sources']={pose:dict(sequence_path=pose+'/smpl_params.npz',frame_index=i,sequence_sha256='b'*64)
        for i,pose in enumerate(sorted(m.deployment.POSE_TEMPLATES))}
    for template in authority['templates'].values():
        template['patient_asset']['identity_split']=str(m.runtime.shared.contract.IDENTITY_SPLIT_PATH)
    specs=deepcopy(old['patient_specs'])
    for row in old['layout']['bodies']:
        spec=specs[row['subject_id']];path=Path(spec['directory'])/'asset_manifest.json'
        value=m.runtime.read_bound(dict(path=str(path),sha256=spec['manifest_sha256']))
        value[0]['source_pose']=authority['pose_sources'][row['pose_template']]
        path.unlink();spec['manifest_sha256']=m.runtime.write_new(path,value)['sha256']
    monkeypatch.setattr(m,'_verified_prerequisites',lambda *a:deepcopy(authority))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata plan opened body arrays'))
    frozen=m.freeze_body_plan(old['admission'],old['bridge_smokes'],seed=old['seed'],patient_specs=specs,
        root=old['layout']['root'],output=tmp/'structured-body-plan.json')
    assert m.verify_body_plan(frozen)['pose_sources']==authority['pose_sources']
    specs[old['layout']['bodies'][0]['subject_id']]['anchor_inventory']=str(tmp/'untrusted-body.npz')
    with pytest.raises(ValueError):m.freeze_body_plan(old['admission'],old['bridge_smokes'],seed=old['seed'],patient_specs=specs,
        root=old['layout']['root'],output=tmp/'changed-anchor-plan.json')


@pytest.mark.parametrize('fault',['physics','patient','pose','seed','extra_route','existing_output'])
def test_preparation_freeze_refuses_control_or_identity_drift(sealed_metadata,fault):
    m,body,configs,tmp_path=sealed_metadata
    if fault=='physics':configs[0]['physics']['unchanged']=2
    elif fault=='patient':configs[0]['patient_asset']['file']='alias-body.npz'
    elif fault=='pose':configs[0][m.ROUTE_KEY]['pose_template']='unseen'
    elif fault=='seed':configs[0][m.ROUTE_KEY]['seed']+=1
    elif fault=='extra_route':configs[0][m.ROUTE_KEY]['allow_test']=True
    else:
        Path(m.runtime.read_bound(body)['layout']['bodies'][0]['prepared_output']).mkdir(parents=True)
    with pytest.raises(ValueError):m.freeze_preparation_plan(body,configs,output=tmp_path/'preparation-plan.json')


@pytest.mark.parametrize('fault',['none','delegate_error','changed_config','stage','order'])
def test_owned_cpu_preparation_uses_explicit_flag_once_and_retains_all34_outcome_slots(sealed_metadata,monkeypatch,fault):
    import sys
    from sim.ausc_policy import geometry_qualification,process_guard
    m,body,configs,tmp=sealed_metadata
    plan=m.freeze_preparation_plan(body,configs,output=tmp/'prep-plan.json')
    issued=m.issue_preparation_capability(plan,output=tmp/'prep-cap.json');ref=issued['owner_configuration']
    config=m.runtime.read_bound(ref);route=config[m.AGGREGATE_KEY];output=Path(route['owner']['review']);output.parent.mkdir(parents=True)
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(process_guard,'process_identity',lambda pid:dict(pid=pid,pgid=1,sid=1,start_ticks='1',state='R'))
    original_bound=m.runtime.bound_path;geometry_calls=[]
    def bound(value):
        if value.get('path')=='/synthetic/original.usdc':geometry_calls.append(value);return tmp/'synthetic-source-only'
        if value.get('sha256')=='a'*64 and Path(value['path']).name=='body.npz':return tmp/'synthetic-body-only'
        return original_bound(value)
    monkeypatch.setattr(m.runtime,'bound_path',bound);calls=[]
    original_error=RuntimeError('original CPU authoring failure')
    def delegate():
        assert all(original_bound(row['config']).is_file() for row in route['slots'])
        assert sys.argv[1:3]==['--cpu-prepare-only','--config']
        calls.append(list(sys.argv));target=Path(sys.argv[-1]);target.mkdir(parents=True)
        if fault=='delegate_error':raise original_error
        m.runtime.write_new(target/'COMPLETE.json',dict(completed=True,physics_executed=False,physical_validation=False,
            training_eligible=False,preparation_backend='cpu_usd_authoring_no_simulation_app'))
        if fault=='changed_config':
            path=Path(route['slots'][0]['config']['path']);content=path.read_bytes();path.write_bytes(content[:-1]+b' ')
    monkeypatch.setattr(geometry_qualification,'main',delegate)
    if fault in ('stage','order'):
        changed=deepcopy(config)
        if fault=='stage':changed[m.AGGREGATE_KEY]['stage']='native'
        else:changed[m.AGGREGATE_KEY]['slots'][:2]=list(reversed(changed[m.AGGREGATE_KEY]['slots'][:2]))
        with pytest.raises(ValueError):m.planning_aggregate_preflight(changed,output=output)
        assert not calls and not output.exists();return
    if fault=='none':
        complete=m.runtime.read_bound(m.run_planning_aggregate(ref,output))
        assert complete['stage']=='preparation' and len(complete['slots'])==len(calls)==34
        assert all('--cpu-prepare-only' in row for row in calls) and len(geometry_calls)>=68
    else:
        with pytest.raises((ValueError,RuntimeError)) as caught:m.run_planning_aggregate(ref,output)
        if fault=='delegate_error':assert caught.value is original_error
        assert len(calls)==1 and not (output/'COMPLETE.json').exists()
    execution=m.runtime.read_bound(m.runtime.reference(output/'preparation_execution.json'))
    assert len(execution['slots'])==34
    assert execution['slots'][0]['status']==('completed' if fault=='none' else 'failed')
    if fault!='none':assert all(row['status']=='planned_unlaunched' for row in execution['slots'][1:])
    else:assert execution['completed'] is True


def test_patient_test_split_without_issued_capability_refuses_before_first_read(monkeypatch):
    from sim.ausc_policy import patient_scene
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID,GEOMETRY_SOURCE_SHA256
    monkeypatch.setattr(patient_scene,'_read',lambda *a:pytest.fail('unissued TEST body/metadata read'))
    with pytest.raises(ValueError,match='capability'):
        patient_scene.load_candidate(dict(expected_split='test',directory='/synthetic/sealed',manifest_sha256='0'*64,geometry_id=GEOMETRY_ID,
            geometry_source_sha256=GEOMETRY_SOURCE_SHA256,freshly_generated=True))


def test_preparation_patient_capability_authenticates_exact_spec_and_stage(sealed_metadata,monkeypatch):
    m,body,configs,tmp=sealed_metadata
    plan=m.freeze_preparation_plan(body,configs,output=tmp/'preparation-plan.json')
    issued=m.issue_preparation_capability(plan,output=tmp/'cap.json')
    config=m.runtime.read_bound(issued['configurations'][0]['config']);spec=config['patient_asset']
    assert m.authorize_patient(spec,config)['subject_id']==config[m.ROUTE_KEY]['subject_id']
    for altered in (dict(spec,file='alias.npz'),dict(spec,expected_split='train')):
        with pytest.raises(ValueError):m.authorize_patient(altered,config)
    changed=deepcopy(config);changed[m.ROUTE_KEY]['schema']=m.NATIVE_ROUTE_SCHEMA
    with pytest.raises(ValueError):m.authorize_patient(spec,changed)


def test_geometry_entry_rejects_unissued_stage_before_backend_or_scene(tmp_path,monkeypatch):
    from sim.ausc_policy import geometry_qualification as geometry
    from sim.ausc_policy.geometry_epoch import GEOMETRY_SOURCE_SHA256
    import sys
    m=module();config=dict(geometry_id=geometry.GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        freshly_generated=True,**{m.ROUTE_KEY:dict(schema=m.PREP_ROUTE_SCHEMA,
            capability=dict(path=str(tmp_path/'unissued.json'),sha256='0'*64))})
    path=m.runtime.write_new(tmp_path/'config.json',config)
    monkeypatch.setattr(sys,'argv',['geometry','--config',path['path'],'--output',str(tmp_path/'absent')])
    monkeypatch.setattr(geometry,'preparation_backend',lambda *a:pytest.fail('backend before sealed authorization'))
    with pytest.raises((ValueError,FileNotFoundError)):geometry.main()


@pytest.fixture
def prepared_metadata(sealed_metadata,monkeypatch):
    m,body,configs,tmp=sealed_metadata
    prep=m.freeze_preparation_plan(body,configs,output=tmp/'prep-plan.json')
    issued=m.issue_preparation_capability(prep,output=tmp/'prep-cap.json')
    # Actual original preparation owner authentication is tested separately.
    monkeypatch.setattr(m,'_preparation_owner',lambda *a:dict(path='/synthetic/preparation-owner.json',sha256='a'*64))
    for declaration in issued['configurations']:
        config=m.runtime.read_bound(declaration['config']);root=Path(declaration['output']);root.mkdir(parents=True)
        (root/'geometry').mkdir()
        scene=m.runtime.write_new(root/'candidate_physics.usdc',{'synthetic_binary_fixture':True})
        collision=m.runtime.write_new(root/'geometry/collision_geometry.json',{'synthetic_fixture':True})
        arrays=m.runtime.write_new(root/'geometry/collision_geometry.npz',{'synthetic_binary_fixture':True})
        for name in ('mass_estimates.json','physics_profile.json','approved_camera_restoration.json'):
            m.runtime.write_new(root/name,{'synthetic_fixture_only':True})
        patient=dict(identity=declaration['subject_id'],split='test',asset_sha256='a'*64,
            identity_split_sha256=m.runtime.shared.contract.IDENTITY_SPLIT_SHA256,
            non_patient_scene_unchanged=True,targets=[dict(name=target,position_world_m=[0.,0.,0.],
                normal_world=[0.,0.,1.],contact_plane_normal_world=[0.,0.,1.],contact_plane_radius_m=.023) for target in m.TARGETS])
        m.runtime.write_new(root/'patient_receipt.json',patient)
        m.runtime.write_new(root/'COMPLETE.json',dict(completed=True,patient=patient,physics_executed=False,
            physical_validation=False,training_eligible=False,preparation_backend='cpu_usd_authoring_no_simulation_app',effective_scene_sha256=scene['sha256'],
            geometry_id=m.runtime.geometry.GEOMETRY_ID,source_sha256=m.runtime.geometry.GEOMETRY_SOURCE_SHA256,
            world_from_left_base=np.eye(4).tolist(),geometry_artifacts=dict(collision_geometry_sha256=collision['sha256'],
                collision_arrays_sha256=arrays['sha256'])))
    return m,body,issued,tmp


def test_prepared_inventory_binds_all34_closed_outputs_before_planning(prepared_metadata):
    m,body,issued,tmp=prepared_metadata
    frozen=m.freeze_prepared_inventory(issued['capability'],output=tmp/'prepared.json')
    value=m.verify_prepared_inventory(frozen)
    assert len(value['bodies'])==34 and value['preparation_capability']==issued['capability']
    row=value['bodies'][0];context=dict(capability=issued['capability'],prepared_inventory=frozen,subject_id=row['subject_id'])
    assert m.authorize_prepared(context,row['preparation_config'],row['prepared_complete'],row['patient_receipt'])==row
    alias=dict(row['patient_receipt'],path=str(tmp/'same-content-alias.json'))
    with pytest.raises(ValueError):m.authorize_prepared(context,row['preparation_config'],row['prepared_complete'],alias)


@pytest.mark.parametrize('fault',['mutated','same_length','added','deleted','link','renamed','empty_directory'])
def test_prepared_closed_inventory_refuses_after_freeze_drift(prepared_metadata,fault):
    m,body,issued,tmp=prepared_metadata
    frozen=m.freeze_prepared_inventory(issued['capability'],output=tmp/'prepared.json')
    root=Path(issued['configurations'][0]['output']);path=root/'candidate_physics.usdc'
    if fault=='mutated':path.write_text('changed')
    elif fault=='same_length':
        m.verify_prepared_inventory(frozen)  # A previous public call never makes later payloads trusted.
        content=path.read_bytes();path.write_bytes(bytes([content[0]^1])+content[1:])
    elif fault=='added':(root/'extra.bin').write_bytes(b'x')
    elif fault=='deleted':path.unlink()
    elif fault=='link':(root/'link').symlink_to(path)
    elif fault=='empty_directory':(root/'extra-empty').mkdir()
    else:path.rename(root/'alias.usdc')
    with pytest.raises(ValueError):m.verify_prepared_inventory(frozen)


def test_builder_rejects_unissued_prepared_capability_before_bound_metadata(monkeypatch):
    from sim.ausc_policy import m2_pilot_config as builder
    monkeypatch.setattr(builder,'bound_json',lambda *a:pytest.fail('unissued builder read before authorization'))
    with pytest.raises((ValueError,FileNotFoundError)):
        builder._prepared({}, {}, {},formal_preparation={'capability':dict(path='/synthetic/unissued.json',sha256='0'*64)})


def test_planning_wrapper_freezes_all340_configs_and_original_bounded_controls(prepared_metadata,monkeypatch):
    m,body,issued,tmp=prepared_metadata
    inventory=m.freeze_prepared_inventory(issued['capability'],output=tmp/'prepared.json')
    aggregate_ref=m.freeze_planning_configurations(inventory)
    aggregate=m.runtime.read_bound(aggregate_ref);configs=aggregate[m.AGGREGATE_KEY]['slots']
    assert len(configs)==340
    layout=m.runtime.read_bound(body)['layout'];owner=layout['planning_owner']
    assert aggregate_ref['path']==owner['config'] and aggregate['base']==owner['base']
    assert not Path(owner['output']).exists()
    assert all(not Path(row['config']['path']).exists() for row in configs)
    assert m.planning_aggregate_preflight(aggregate,output=owner['review'])['layout']==layout
    monkeypatch.setattr(np,'load',lambda *a:pytest.fail('planning metadata preflight decoded payload'))
    for row in (configs[0],configs[-1]):
        config=row['configuration'];binding=m.planning_preflight(config,output=row['output'])
        assert config['seeds_per_candidate']==12 and config['per_gap_seconds']==180.
        assert config['plane_tilt_limit_deg']==3. and config['fresh_pre_gap_search_m']==[.025,.03]
        assert config['loaded_compression_m']==.0025 and config['collision_profile']=='full'
        assert binding['subject_id']==row['subject_id'] and binding['target_id']==row['target_id']
        for fault in ('seed','target','output','source','capability'):
            changed=deepcopy(config)
            if fault=='seed':changed['seeds_per_candidate']=13
            elif fault=='target':changed['targets'][0]['position_world_m'][0]=1.
            elif fault=='output':changed[m.PLANNING_KEY]['output']+='_alias'
            elif fault=='source':changed['geometry_source']+='_alias'
            else:changed[m.PLANNING_KEY]['capability']=dict(path=str(tmp/'unissued'),sha256='0'*64)
            with pytest.raises((ValueError,FileNotFoundError)):m.planning_preflight(changed,output=row['output'])
    for fault in ('missing','extra','reordered','changed','alias'):
        changed=deepcopy(aggregate);slots=changed[m.AGGREGATE_KEY]['slots']
        if fault=='missing':slots.pop()
        elif fault=='extra':slots.append(deepcopy(slots[0]))
        elif fault=='reordered':slots[:2]=list(reversed(slots[:2]))
        elif fault=='changed':slots[0]['configuration']['seeds_per_candidate']=13
        else:slots[0]['config']['path']+='_alias'
        with pytest.raises(ValueError):m.planning_aggregate_preflight(changed,output=owner['review'])


@pytest.mark.parametrize('fault',['none','existing','link','changed_slot'])
def test_aggregate_materializes_all_frozen_slot_bytes_before_first_delegate(prepared_metadata,monkeypatch,fault):
    import json,sys
    from sim.ausc_policy import process_guard
    m,body,issued,tmp=prepared_metadata
    inventory=m.freeze_prepared_inventory(issued['capability'],output=tmp/'prepared.json')
    aggregate_ref=m.freeze_planning_configurations(inventory);aggregate=m.runtime.read_bound(aggregate_ref)
    route=aggregate[m.AGGREGATE_KEY];output=Path(route['owner']['review']);output.parent.mkdir(parents=True)
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(process_guard,'process_identity',lambda pid:dict(pid=pid,pgid=1,sid=1,start_ticks='1',state='R'))
    calls=[]
    def delegate(config_ref,path):
        # All 340 exact refs must exist before the first unchanged delegate.
        if not calls:assert all(m.runtime.bound_path(row['config']).is_file() for row in route['slots'])
        calls.append(config_ref);Path(path).mkdir(parents=True)
        receipt=m.runtime.write_new(Path(path)/'heldout_planning.json',{'synthetic_delegate_only':True})
        if fault=='changed_slot':Path(route['slots'][-1]['config']['path']).write_text('{}')
        return receipt
    monkeypatch.setattr(m,'run_planning',delegate)
    if fault=='existing':output.mkdir()
    elif fault=='link':output.symlink_to(tmp/'absent-output')
    if fault=='none':
        result=m.run_planning_aggregate(aggregate_ref,output);value=m.runtime.read_bound(result)
        assert len(calls)==len(value['slots'])==340 and value['configuration']==aggregate_ref
        assert value['source_inventory']==m.planning_sources() and value['physics_executed'] is False
    else:
        with pytest.raises((ValueError,FileExistsError)):m.run_planning_aggregate(aggregate_ref,output)
        assert len(calls)==(339 if fault=='changed_slot' else 0)
        assert not (output/'COMPLETE.json').exists()


def test_tool_formal_dispatch_authenticates_before_assets_output_or_spawn(tmp_path,monkeypatch):
    import json,sys
    from sim.ausc_policy import tool_geometry_review,isaac_case
    m=module();config=dict(experiment='m2_geometry_plan',**{m.AGGREGATE_KEY:dict(schema=m.AGGREGATE_SCHEMA)})
    path=tmp_path/'unissued.json';path.write_text(json.dumps(config));output=tmp_path/'output'
    monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(sys,'argv',['tool','--config',str(path),'--output',str(output)])
    monkeypatch.setattr(isaac_case,'approved_assets',lambda *a:pytest.fail('assets before aggregate authorization'))
    monkeypatch.setattr(tool_geometry_review.subprocess,'Popen',lambda *a,**k:pytest.fail('spawn before aggregate authorization'))
    with pytest.raises((ValueError,KeyError)):tool_geometry_review.main()
    assert not output.exists()


def test_owned_evaluation_cli_rejects_unissued_aggregate_without_delegate_or_output(tmp_path,monkeypatch):
    import sys
    from sim.ausc_policy import geometry_plan
    m=module();config=m.runtime.write_new(tmp_path/'unissued-owner.json',dict(experiment='m2_geometry_plan',
        **{m.AGGREGATE_KEY:dict(schema=m.AGGREGATE_SCHEMA)}));output=tmp_path/'review'
    monkeypatch.setattr(sys,'argv',['evaluation','--config',config['path'],'--output',str(output)])
    monkeypatch.setattr(geometry_plan,'main',lambda:pytest.fail('bare planner before sealed authorization'))
    with pytest.raises(ValueError):m.main()
    assert not output.exists()


@pytest.mark.parametrize('fault',['none','identity','source','owner_commit','worker_command','missing_slot','extra_slot','stage'])
def test_cpu_aggregate_owner_requires_original_guard_child_identity_and_all_actual_receipts(tmp_path,monkeypatch,fault):
    from sim.ausc_policy import process_guard
    m=module();root=tmp_path/'runs/ausc_policy/frozen-cpu';out=root/'output';review=out/'review';review.mkdir(parents=True)
    slots=[];actual=[];identity=dict(pid=101,pgid=100,sid=100,start_ticks='2',state='R')
    for i in range(34):
        subject='p%06d'%i;slot=review/'prepared'/subject
        receipt=m.runtime.write_new(slot/'heldout_preparation.json',{'synthetic_only':True})
        slots.append(dict(subject_id=subject,output=str(slot)));actual.append(dict(subject_id=subject,receipt=receipt))
    owner=dict(run_id=root.name,output=str(out),review=str(review),config=str(tmp_path/'owner.json'))
    cfg=m.runtime.write_new(owner['config'],dict(experiment='m2_geometry_plan',isaac_python='/synthetic/python',
        **{m.AGGREGATE_KEY:dict(stage='preparation',owner=owner,slots=slots,prepared_inventory=None)}))
    source=m.runtime.reference(process_guard.__file__)
    command=[m.runtime.sys.executable,'-u','-B','-m','sim.ausc_policy.tool_geometry_review','--config',cfg['path'],'--output',str(out)]
    record=dict(run_id=root.name,code_commit='a'*40,config=m.runtime.read_bound(cfg),config_sha256=cfg['sha256'],command=command,
        resource_guard=dict(path=source['path'],sha256=source['sha256'],gpu_exit_grace_s=process_guard.EXIT_GPU_GRACE_S),started_utc='fixed')
    stage=dict(passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[],
        process_identity=dict(pid=100,pgid=100,sid=100,start_ticks='1',members={'100':'1','101':'2'}))
    if fault=='owner_commit':record['code_commit']='b'*40
    m.runtime.write_new(root/'run.json',record)
    terminal=m.runtime.write_new(root/'result.json',dict(record,stage=stage,completed=True,gpu_released=True))
    complete=dict(schema=m.AGGREGATE_SCHEMA,stage='preparation',configuration=cfg,prepared_inventory=None,
        source_inventory=m.planning_sources(),identity=identity,slots=actual,completed=True,physics_executed=False,training_eligible=False,wall_s=1.)
    if fault=='missing_slot':complete['slots']=actual[:-1]
    elif fault=='extra_slot':complete['slots']=actual+[actual[0]]
    elif fault=='stage':complete['stage']='planning'
    complete_ref=m.runtime.write_new(review/'COMPLETE.json',complete)
    m.runtime.write_new(out/'COMPLETE.json',dict(complete,isaac_exit_code=0))
    worker=dict(worker=m.MODULE,pid=101,identity=dict(identity,state='S'),returncode=0,clean_exit=True,timed_out=False,
        complete_file_present=True,configuration=cfg,command=['/synthetic/python','-u','-B','-m',m.MODULE,'--config',cfg['path'],'--output',str(review)],
        output=str(review),source_inventory=m.planning_sources(),complete=complete_ref)
    if fault=='identity':worker['identity']['start_ticks']='3'
    elif fault=='source':worker['source_inventory']={}
    elif fault=='worker_command':worker['command'][4]='sim.ausc_policy.geometry_plan'
    m.runtime.write_new(out/'native_worker_exit.json',worker)
    processing=dict(processing_freeze=m.runtime.write_new(tmp_path/'source.json',{'reviewed_commit':'a'*40}))
    monkeypatch.setattr(m,'planning_aggregate_preflight',lambda *a,**kw:None)
    if fault=='none':assert m._aggregate_owner(terminal,cfg,dict(identity=identity),processing,'preparation')['slots']==actual
    else:
        with pytest.raises(ValueError):m._aggregate_owner(terminal,cfg,dict(identity=identity),processing,'preparation')


def test_planning_source_binding_also_works_for_python_module_main(monkeypatch):
    import sys
    m=module();monkeypatch.delitem(sys.modules,m.MODULE)
    assert m.planning_sources()[m.MODULE]==m.runtime.reference(m.__file__)


def test_authority_verification_reuses_only_completed_work_inside_one_call_and_never_across_calls(tmp_path,monkeypatch):
    m=module();admission=m.runtime.write_new(tmp_path/'admission.json',{});smokes=m.runtime.write_new(tmp_path/'smokes.json',{});calls=[]
    def checked(a,s):calls.append((a,s));return dict(observation=len(calls))
    monkeypatch.setattr(m,'_verified_prerequisites',checked,raising=False)
    @m._verification_scope
    def operation():
        first=m._prerequisites(admission,smokes);first['observation']=100
        return m._prerequisites(admission,smokes)
    assert operation()=={'observation':1} and len(calls)==1
    assert operation()=={'observation':2} and len(calls)==2
    def refuse(a,s):raise ValueError('changed immutable source')
    monkeypatch.setattr(m,'_verified_prerequisites',refuse)
    with pytest.raises(ValueError,match='changed immutable source'):operation()


def test_attached_native_audit_is_one_original_tick_and_default_route_is_untouched(tmp_path,monkeypatch):
    m,unused,c,state=native_tick_fixture(tmp_path);unused.close();calls=[]
    def tick(command,subphase,active=False,recovery=False):
        calls.append((active,recovery));state['time_s']+=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=[0.]*7,q_command=list(command),phase='FREE',subphase=subphase,
            physical_violations=[],target='L1'))
    c._tick=tick
    assert m.attach_native_audit(c,{},tmp_path) is None and c._tick is tick
    permit=dict(processing={});row=dict(metadata=c.metadata)
    monkeypatch.setattr(m.deployment,'preflight',lambda *a,**k:dict(permit=permit,attempt=row))
    monkeypatch.setattr(m,'_sources',lambda *a:{})
    audit=m.attach_native_audit(c,{'m2_given_contact_deployment':{}},tmp_path)
    c._tick([0.]*7,'staging',active=True)
    assert calls==[(True,False)] and audit.rows==1
    receipt=m.close_native_audit(c)
    assert receipt['complete'] is True and c._tick is tick
    import gzip,json
    with gzip.open(receipt['trace']['path'],'rt') as stream:event=json.loads(stream.readline())
    assert event['native_row']['target']=='L1' and event['observed_after']['physics_row']==1


def test_neutral_audit_finalizer_failure_does_not_mask_original_physics_exception(tmp_path,monkeypatch):
    m,unused,c,state=native_tick_fixture(tmp_path);unused.close()
    class OriginalError(RuntimeError):pass
    original=OriginalError('original actual physical gate')
    def tick(command,subphase,**kw):raise original
    c._tick=tick
    monkeypatch.setattr(m.deployment,'preflight',lambda *a,**k:dict(permit={'processing':{}},attempt={'metadata':c.metadata}))
    monkeypatch.setattr(m,'_sources',lambda *a:{})
    audit=m.attach_native_audit(c,{'m2_given_contact_deployment':{}},tmp_path)
    monkeypatch.setattr(audit,'_append',lambda *a:(_ for _ in ()).throw(OSError('audit disk error')))
    with pytest.raises(OriginalError) as caught:c._tick([0.]*7,'staging')
    assert caught.value is original and audit.errors
    m.close_native_audit(c)


def act_audit_fixture(tmp_path):
    from test_ausc_given_contact_deployment import request_fixture
    _,ref,dep,permit,request=request_fixture(tmp_path);m=module()
    from sim.ausc_policy.m2_given_contact_intent import IntentDispatcher
    trajectory=SimpleNamespace(input=SimpleNamespace(target_position=[0.]*7))
    trajectory.set_target=lambda q:setattr(trajectory.input,'target_position',list(q))
    setter=IntentDispatcher(trajectory,reference_sha256=ref.sha256);events=[]
    for i in range(24):
        subphase='given_l2_history_wait' if i<12 else 'given_l2_motion'
        if i%12==0:setter.issue([0.]*7,time_s=i/240.,physics_row=i,context=request['context']['context'],owner='ACT_L2',role=subphase)
        raw=setter.snapshot(time_s=i/240.,physics_row=i);raw['live_target_after_step_q']=[0.]*7
        labels=dict(phase='FREE',subphase=subphase,actual_control_owner='ACT_L2',actor={'phase':'FREE'},raw_intent=raw)
        def observed(index):return dict(physics_row=index,time_s=index/240.,q=[0.]*7,pose_base=np.eye(4).tolist())
        events.append(dict(physics_row=i,time_s=(i+1)/240.,observed_before=observed(i),observed_after=observed(i+1),
            controller_before=deepcopy(labels),controller_after=deepcopy(labels),actual_control_owner='ACT_L2',raw_intent=raw,
            q_command=[0.]*7))
    response=m.deployment.response_for(request,np.zeros((20,7)))
    ipc=[dict(event='inference',request=request,response=response,accepted=True,request_id=1,simulated_time_pause_s=0.)]
    run=dict(l2_queries=[dict(index=0,physics_row=12,time_s=.05,accepted=True,
        history=request['history'],history_times=request['history_times'],absolute_h20=[[0.]*7]*20)],
        l2_actions=[dict(query_index=0,prefix_index=0,physics_row_start=12,time_s=.05,proposal=[0.]*7,completed=True)])
    return m,events,ipc,run,dep,permit


def test_act_audit_reconstructs_every_observation_prediction_and_actual_e10_prefix(tmp_path):
    m,events,ipc,run,dep,permit=act_audit_fixture(tmp_path);calls=[]
    def predictor(*args,**kwargs):calls.append(deepcopy(kwargs));return np.zeros((20,7))
    result=m._audit_act_execution(events,ipc,run,dep,permit,predictor)
    assert result['inferences']==1 and result['executed_actions']==1 and result['executed_native_ticks']==12
    assert result['prediction_max_abs_diff']==0 and len(calls)==1


@pytest.mark.parametrize('returned',[True,False])
def test_failed_act_still_reconstructs_returned_chunk_without_inventing_an_executed_action(tmp_path,returned):
    m,events,ipc,run,dep,permit=act_audit_fixture(tmp_path);events=events[:12]
    run.update(success=False,l2_actions=[]);run['l2_queries'][0].update(accepted=False,failure_reason='original rejected chunk')
    run['l2_queries'][0].pop('absolute_h20');calls=[]
    if not returned:
        ipc[0].pop('response');ipc[0].update(accepted=False,error='original IPC timeout')
    def predictor(*args,**kwargs):calls.append(1);return np.zeros((20,7))
    result=m._audit_act_execution(events,ipc,run,dep,permit,predictor)
    assert result['inferences']==1 and result['returned_chunks']==int(returned) and len(calls)==int(returned)
    assert result['executed_actions']==result['executed_native_ticks']==0
    assert result['failed_queries']==1 and result['action_ranges']==[]
    run['success']=True
    with pytest.raises(ValueError):m._audit_act_execution(events,ipc,run,dep,permit,predictor)


@pytest.mark.parametrize('fault',['history','response','origin','prefix','proposal','native_owner','native_target','outside_l2','missing_action','duplicate_action'])
def test_act_audit_refuses_forged_predictions_history_or_native_attribution(tmp_path,fault):
    m,events,ipc,run,dep,permit=act_audit_fixture(tmp_path)
    if fault=='history':ipc[0]['request']['history'][0][0]=.01
    elif fault=='response':ipc[0]['response']['relative'][19][6]=.01
    elif fault=='origin':ipc[0]['request']['measured_q'][0]=.01
    elif fault=='prefix':run['l2_actions'][0]['prefix_index']=10
    elif fault=='proposal':run['l2_actions'][0]['proposal'][0]=.01
    elif fault=='native_owner':events[12]['actual_control_owner']='TEACHER_L2'
    elif fault=='native_target':events[12]['raw_intent']['live_target_q'][0]=.01
    elif fault=='outside_l2':events[12]['controller_before']['subphase']='L3_ALIGN'
    elif fault=='missing_action':run['l2_actions']=[]
    else:run['l2_actions'].append(deepcopy(run['l2_actions'][0]))
    with pytest.raises(ValueError):m._audit_act_execution(events,ipc,run,dep,permit,lambda *a,**k:np.zeros((20,7)))


@pytest.mark.parametrize('value',[{}, {'schema':'pilot','passed':True},
    {'schema':'m2_given_contact_actual_bridge_smoke_qualification_v1','passed':True,'cpu_fixture_only':True}])
def test_self_reported_or_cpu_smokes_cannot_issue_test_authority(tmp_path,value,monkeypatch):
    m=module();ref=m.runtime.write_new(tmp_path/'fake-smoke.json',value)
    monkeypatch.setattr(m.deployment,'_admission_metadata',lambda *a:pytest.fail('unqualified schema reached model/source reads'))
    with pytest.raises(ValueError):m.verify_bridge_smokes(ref)


def test_native_trace_reconstruction_rejects_changed_actual_pose_or_row(tmp_path):
    m,audit,c,state=native_tick_fixture(tmp_path)
    def tick():
        state['time_s']=1/240.
        c.env.rows.append(dict(time_s=state['time_s'],q=[0.]*7,q_command=[0.]*7,phase='FREE',subphase='staging',physical_violations=[]))
    audit.tick(c,[0.]*7,'staging',tick);receipt=audit.close()
    assert len(m.read_native_trace(receipt,c.metadata))==1
    import json
    path=Path(receipt['trace']['path']);event=json.loads(path.read_text())
    event['observed_after']['pose_base'][0][0]=2.
    path.write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError):m.read_native_trace(receipt,c.metadata)


@pytest.mark.parametrize('fault',['none','short_release','relaxed_plane','align_budget','coarse_gate'])
def test_heldout_physical_auditor_preserves_original_gates_without_relabeling(tmp_path,monkeypatch,fault):
    # Numerical fixtures only. Native parking replay has its separate real
    # Ruckig proof; no physical qualification or capability is emitted here.
    from test_ausc_given_contact_dataset import physical_fixture
    run,audit,meta=physical_fixture(tmp_path,monkeypatch,release_ticks=36 if fault=='short_release' else 48)
    meta['split']='test';m=module()
    if fault=='relaxed_plane':run['contact_plane_contracts'][meta['target_id']]['tilt_relaxed']=True
    elif fault=='align_budget':run['l2_handoffs'][0]['alignment_actual_audit']['observed']['max_local_plane_deg']=10.
    elif fault=='coarse_gate':run['phase_config']['precontact_tolerance_m']=.02
    if fault=='none':m._physical_scalars(run,audit,meta)
    else:
        with pytest.raises(ValueError):m._physical_scalars(run,audit,meta)
    assert meta['split']=='test'


def test_scene_formal_capability_precedes_binding_payload_and_authoring_strip(monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();config={m.ROUTE_KEY:dict(schema=m.NATIVE_ROUTE_SCHEMA,
        capability=dict(path='/synthetic/unissued-native.json',sha256='0'*64))}
    monkeypatch.setattr(scene_store,'read_source',lambda *a,**k:pytest.fail('scene payload before native capability'))
    for operation in (lambda:scene_store.validate_config(config),lambda:scene_store._authoring(config)):
        with pytest.raises((ValueError,FileNotFoundError)):operation()


def test_stored_scene_native_capability_authenticates_without_original_config(monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();stored=dict(schema='m2_given_contact_sealed_test_scene_ref_v1',binding={},scene={},
        authorization=dict(schema='m2_given_contact_sealed_test_scene_authorization_v1',
            capability=dict(path='/synthetic/unissued-native.json',sha256='0'*64),episode_id='test-episode',configuration_sha256='a'*64))
    monkeypatch.setattr(scene_store,'real_path',lambda *a,**k:pytest.fail('stored scene reached CAS before native authorization'))
    calls=[]
    def refuse(binding,authorization):calls.append((binding,authorization));raise ValueError('unissued exact native cap')
    monkeypatch.setattr(m,'verify_scene_authorization',refuse,raising=False)
    with pytest.raises(ValueError,match='unissued exact native cap'):scene_store.resolve_scene(stored)
    assert calls==[(stored['binding'],stored['authorization'])]


def test_native_scene_publication_refuses_before_creating_any_cas_path(tmp_path,monkeypatch):
    from sim.ausc_policy import scene_store
    m=module();root=tmp_path/'not-created-cas'
    binding=dict(store_root=str(root),sources=dict(scene=dict(file='/synthetic/never-open.usdc',
        sha256='a'*64,bytes=1)))
    def refuse(*args):raise ValueError('unissued publication capability')
    monkeypatch.setattr(m,'verify_scene_authorization',refuse)
    with pytest.raises(ValueError,match='unissued publication capability'):
        scene_store._publish(binding,authorization={})
    assert not root.exists()


def test_formal_deployment_dispatches_only_new_native_schema_to_evaluation(tmp_path,monkeypatch):
    m=module();ref=m.runtime.write_new(tmp_path/'native-cap.json',dict(schema=m.CAPABILITY_SCHEMA))
    calls=[]
    def verify(value,**kwargs):calls.append((value,kwargs));raise ValueError('unissued native authority')
    monkeypatch.setattr(m,'verify_capability',verify)
    for operation in (lambda:m.deployment.verify_permit(ref),lambda:m.deployment._child_permit(ref)):
        with pytest.raises(ValueError,match='unissued native authority'):operation()
    assert calls==[(ref,dict(stage='native')),(ref,dict(stage='native',child_only=True))]


def test_test_request_never_uses_membership_flag_or_unverified_old_permit(tmp_path):
    from test_ausc_given_contact_deployment import request_fixture
    _,ref,dep,permit,request=request_fixture(tmp_path);m=module()
    request['metadata']['split']='test';dep['metadata']['split']='test';permit['attempts'][0]['metadata']['split']='test'
    permit['attempts'][0]['split']='test'
    with pytest.raises(ValueError):m.deployment.validate_request(request,dep,permit,previous=None)


def test_independent_align_uses_actual_full_rotation_and_never_command_fk_as_measurement(tmp_path):
    from test_ausc_given_contact_pose import reference
    from scipy.spatial.transform import Rotation
    m=module();ref=reference(tmp_path);pose=ref.pose_base.copy();pose[:3,3]-=pose[:3,2]*.03
    plane=dict(contact_plane_normal_world=[0.,0.,-1.],plane_tilt_limit_deg=3.,contact_plane_radius_m=.023,tilt_relaxed=False)
    kin=SimpleNamespace(tcp=lambda q:pose.copy())
    event=dict(q_command=[0.]*7,controller_after={'subphase':'L3_ALIGN'},
        observed_before={'pose_base':pose.tolist()},observed_after={'pose_base':pose.tolist()})
    report=m._alignment_actual([event],ref,plane,np.eye(4),kin=kin)
    assert report['observed']['samples']==2 and report['command']['samples']==2
    changed=deepcopy(event);actual=pose.copy();actual[:3,:3]=pose[:3,:3]@Rotation.from_euler('z',.2).as_matrix()
    changed['observed_after']['pose_base']=actual.tolist()
    with pytest.raises(ValueError):m._alignment_actual([changed],ref,plane,np.eye(4),kin=kin)


@pytest.fixture
def planned_metadata(prepared_metadata,monkeypatch):
    m,body,issued,tmp=prepared_metadata
    inventory=m.freeze_prepared_inventory(issued['capability'],output=tmp/'prepared.json')
    aggregate=m.runtime.read_bound(m.freeze_planning_configurations(inventory));configs=aggregate[m.AGGREGATE_KEY]['slots']
    prepared=m.verify_prepared_inventory(inventory)
    index={row['subject_id']:row for row in prepared['bodies']};records=[]
    # The real original CPU-owner proof is a separate authority boundary. No
    # actual planner/model/TEST body is run by this synthetic protocol fixture.
    monkeypatch.setattr(m,'_planning_owner',lambda *a,**k:None)
    owner_path=Path(aggregate[m.AGGREGATE_KEY]['owner']['output']).parent/'result.json'
    owner=m.runtime.write_new(owner_path,{'synthetic_fixture_only':True})
    for declaration in configs:
        config=declaration['configuration'];m.runtime.write_new(declaration['config']['path'],config)
        root=Path(declaration['output']);root.mkdir(parents=True)
        target=declaration['target_id'];identity=declaration['subject_id'];row=index[identity]
        patient=m.runtime.read_bound(row['patient_receipt']);point=next(t for t in patient['targets'] if t['name']==target)
        complete=m.runtime.read_bound(row['prepared_complete'])
        body_row=next(b for b in m.runtime.read_bound(body)['layout']['bodies'] if b['subject_id']==identity)
        found=target in body_row['target_priority'][2:4]
        result=m.runtime.write_new(root/'COMPLETE.json',dict(completed=True,geometry_id=m.runtime.geometry.GEOMETRY_ID,
            geometry_artifacts=complete['geometry_artifacts'],physics_executed=False,training_eligible=False,
            targets=[dict(name=target,geometry_path_found=found)]))
        if found:
            patient_ref={'file':row['patient_receipt']['path'],'sha256':row['patient_receipt']['sha256']}
            plan=dict(geometry_id=m.runtime.geometry.GEOMETRY_ID,geometry_source_sha256=m.runtime.geometry.GEOMETRY_SOURCE_SHA256,
                freshly_generated=True,success=True,name=target,position_world=point['position_world_m'],normal_world=point['normal_world'],
                command_rotation_world=np.diag([1.,-1.,-1.]).tolist(),command_normal_world=[0.,0.,1.],
                world_from_left_base=complete['world_from_left_base'],geometry_artifacts=complete['geometry_artifacts'],
                posture_limits=config['posture_limits'],free_approach_start_gap_m=.025,loaded_compression_m=.0025,
                patient_provenance=dict(binding=patient_ref,identity=identity,asset_sha256='a'*64),
                selected_fit=dict(first_contact=dict(contact_plane_normal_world=[0.,0.,1.],contact_plane_radius_m=.023,
                    plane_tilt_limit_deg=3.,tilt_relaxed=False)))
            m.runtime.write_new(root/(target+'_fresh_plan.json'),plan)
        receipt=m.runtime.write_new(root/'heldout_planning.json',dict(schema=m.PLANNING_RESULT_SCHEMA,
            configuration=declaration['config'],prepared_inventory=inventory,output=str(root),complete=result,
            source_inventory=m.planning_sources(),identity=dict(pid=1,pgid=1,sid=1,start_ticks='1',state='R'),completed=True))
        records.append(dict(subject_id=identity,target_id=target,receipt=receipt,owner_terminal=owner))
    return m,inventory,records,tmp


def test_frozen_pair_configs_and_distinct_native_capability_have_no_hash_cycle(planned_metadata):
    m,prepared,records,tmp=planned_metadata
    pair=m.freeze_pair_plan(prepared,records,owner_run_id='synthetic-heldout-owner',output=tmp/'pairs.json')
    value=m.verify_pair_plan(pair)
    assert len(value['planning_results'])==340 and len(value['attempts'])==68
    issued=m.issue_native_capability(pair,output=tmp/'native-cap.json')
    assert len(issued['configurations'])==68
    for declared in (issued['configurations'][0],issued['configurations'][1]):
        config=m.runtime.read_bound(declared['config']);row=m.preflight(config,output=declared['output'])['row']
        assert row['arm']==declared['arm'] and row['split']=='test'
        if row['arm']=='act':assert m.deployment.preflight(config)['attempt']==row
        else:assert 'm2_given_contact_deployment' not in config
        assert 'raw_multimodal_capture' not in config['fresh_expert']
    with pytest.raises(ValueError):m.verify_capability(issued['capability'],stage='preparation')


def test_cpu_child_rebinds_sealed_pair_metadata_without_opening_privileged_plans(planned_metadata,monkeypatch):
    m,prepared,records,tmp=planned_metadata
    pair=m.freeze_pair_plan(prepared,records,owner_run_id='synthetic-heldout-owner',output=tmp/'pairs.json')
    issued=m.issue_native_capability(pair,output=tmp/'native-cap.json');original=m.runtime.reference
    def reference(path):
        name=str(path)
        if name.endswith('_fresh_plan.json') or '/review/slots/' in name:
            pytest.fail('Inference child opened privileged planning/trajectory leaf')
        return original(path)
    monkeypatch.setattr(m.runtime,'reference',reference)
    cap=m.deployment._child_permit(issued['capability'])
    assert len(cap['attempts'])==68 and cap['model']==m.runtime.read_bound(issued['capability'])['model']


def test_native_config_output_and_identity_reject_before_any_prepared_payload(planned_metadata,monkeypatch):
    m,prepared,records,tmp=planned_metadata
    pair=m.freeze_pair_plan(prepared,records,owner_run_id='synthetic-heldout-owner',output=tmp/'pairs.json')
    issued=m.issue_native_capability(pair,output=tmp/'native-cap.json');declared=issued['configurations'][0]
    config=m.runtime.read_bound(declared['config']);original=m._tree
    def tree(root,*,payloads=True):
        if payloads:pytest.fail('Prepared body/USD/collision payload before current config/output/identity authorization')
        return original(root,payloads=False)
    monkeypatch.setattr(m,'_tree',tree)
    for fault in ('physics','extra_field','identity','output'):
        changed=deepcopy(config);output=declared['output']
        if fault=='physics':changed['physics']={'unapproved':True}
        elif fault=='extra_field':changed['unapproved']=True
        elif fault=='identity':changed[m.ROUTE_KEY]['episode_id']='outside-capability'
        else:output=output+'-alias'
        with pytest.raises(ValueError):m.preflight(changed,output=output)
    cap=m.runtime.read_bound(issued['capability']);row=cap['attempts'][0]
    authorization=dict(schema=m.SCENE_AUTH_SCHEMA,capability=issued['capability'],episode_id=row['episode_id'],
        configuration_sha256=row['configuration_sha256'])
    binding=deepcopy(row['configuration']['compact_scene_storage']);binding['batch_id']='unapproved'
    with pytest.raises(ValueError):m.verify_scene_authorization(binding,authorization)


def test_issued_preparation_requires_explicit_cpu_cli_before_backend_or_simulation(sealed_metadata,monkeypatch):
    import sys
    from sim.ausc_policy import geometry_qualification as geometry
    m,body,configs,tmp=sealed_metadata
    plan=m.freeze_preparation_plan(body,configs,output=tmp/'stage-preparation-plan.json')
    issued=m.issue_preparation_capability(plan,output=tmp/'stage-preparation-cap.json');row=issued['configurations'][0]
    class BackendReached(BaseException):pass
    calls=[]
    def backend(*args):calls.append(args[1]);raise BackendReached()
    monkeypatch.setattr(geometry,'preparation_backend',backend)
    command=['geometry','--config',row['config']['path'],'--output',row['output']]
    monkeypatch.setattr(sys,'argv',command)
    with pytest.raises(ValueError):geometry.main()
    assert calls==[] and not Path(row['output']).exists()
    monkeypatch.setattr(sys,'argv',[*command,'--cpu-prepare-only'])
    with pytest.raises(BackendReached):geometry.main()
    assert calls==[True] and not Path(row['output']).exists()


def test_issued_native_rejects_cpu_cli_before_payload_and_accepts_native_cli(planned_metadata,monkeypatch):
    import sys
    from sim.ausc_policy import geometry_qualification as geometry
    m,prepared,records,tmp=planned_metadata
    pair=m.freeze_pair_plan(prepared,records,owner_run_id='synthetic-heldout-owner',output=tmp/'stage-pairs.json')
    issued=m.issue_native_capability(pair,output=tmp/'stage-native-cap.json');row=issued['configurations'][0]
    class BackendReached(BaseException):pass
    calls=[]
    def backend(*args):calls.append(args[1]);raise BackendReached()
    monkeypatch.setattr(geometry,'preparation_backend',backend)
    command=['geometry','--config',row['config']['path'],'--output',row['output']]
    original=m._tree
    def tree(root,*,payloads=True):
        if payloads:pytest.fail('Wrong native CLI opened a prepared payload')
        return original(root,payloads=False)
    monkeypatch.setattr(m,'_tree',tree)
    monkeypatch.setattr(sys,'argv',[*command,'--cpu-prepare-only'])
    with pytest.raises(ValueError):geometry.main()
    assert calls==[] and not Path(row['output']).exists()
    monkeypatch.setattr(m,'_tree',original);monkeypatch.setattr(sys,'argv',command)
    with pytest.raises(BackendReached):geometry.main()
    assert calls==[False] and not Path(row['output']).exists()


def test_patient_split_source_is_exact_authenticated_historical_template_role(tmp_path,monkeypatch):
    m=module();historical=tmp_path/'historic-collector/configs/identity_split.json'
    historical.parent.mkdir(parents=True)
    historical.write_bytes(m.runtime.shared.contract.IDENTITY_SPLIT_PATH.read_bytes())
    body_root=tmp_path/'metadata';body_root.mkdir();pose={'known':'pose'}
    manifest=m.runtime.write_new(body_root/'asset_manifest.json',[dict(file='body.npz',identity='synthetic-id',split='test',
        sha256='a'*64,mesh_scale=1.,numeric_candidate_screen_passed=True,source_pose=pose)])
    summary=m.runtime.write_new(body_root/'summary.json',dict(schema='m2_observed_seated_retarget_v1',
        identity_split_sha256=m.runtime.shared.contract.IDENTITY_SPLIT_SHA256))
    spec=dict(expected_split='test',directory=str(body_root),file='body.npz',identity_split=str(historical),
        manifest_sha256=manifest['sha256'],summary_sha256=summary['sha256'])
    template=dict(spec,expected_split='train')
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Metadata role verification read a body payload'))
    result=m._patient_metadata(spec,'synthetic-id',pose,template)
    assert result['metadata']['identity_split']==m.runtime.reference(historical)
    alias=tmp_path/'same-content-other-role.json';alias.write_bytes(historical.read_bytes())
    with pytest.raises(ValueError):m._patient_metadata(dict(spec,identity_split=str(alias)),'synthetic-id',pose,template)
    historical.write_bytes(historical.read_bytes()+b' ')
    with pytest.raises(ValueError):m._patient_metadata(spec,'synthetic-id',pose,template)


@pytest.mark.parametrize('fault',['missing','reordered','duplicate','wrong_identity','bare_result'])
def test_pair_freeze_refuses_missing_or_unbound_planning_outcomes(planned_metadata,fault):
    m,prepared,records,tmp=planned_metadata
    if fault=='missing':records.pop()
    elif fault=='reordered':records[0],records[1]=records[1],records[0]
    elif fault=='duplicate':records[-1]=records[0]
    elif fault=='wrong_identity':records[0]['subject_id']='p999999'
    else:records[0]['receipt']=m.runtime.read_bound(records[0]['receipt'])['complete']
    with pytest.raises(ValueError):m.freeze_pair_plan(prepared,records,owner_run_id='synthetic-heldout-owner',output=tmp/'pairs.json')


def test_video_receipt_requires_original10hz_actual_native_frame_indices(tmp_path,monkeypatch):
    import json
    m=module();(tmp_path/'actual_fresh_expert.mp4').write_bytes(b'synthetic video bytes; not a real recording')
    frames=[dict(video_frame=i,physics_row=24*(i+1)-1,time_s=24*(i+1)/240.) for i in range(2)]
    path=m.runtime.write_new(tmp_path/'video_frames.json',frames)
    events=[dict(time_s=(i+1)/240.) for i in range(48)]
    run=dict(video=dict(file=str(tmp_path/'actual_fresh_expert.mp4'),frames=2,fps=10,width=640,height=360,camera='/World/RobotCam'))
    monkeypatch.setattr(m,'_probe_video',lambda path:dict(width=640,height=360,r_frame_rate='10/1',nb_frames='2'))
    assert m._video_evidence(tmp_path,run,events)['frame_index']==path
    frames[1]['physics_row']=46
    Path(path['path']).write_text(json.dumps(frames))
    with pytest.raises(ValueError):m._video_evidence(tmp_path,run,events)


def test_zero_available_bodies_still_freezes_original_owner_and34_denominator():
    m,layout=metadata_layout();inventory=m.planning_inventory(layout,planning_records(layout,0))
    schedule=m.paired_schedule(layout,inventory,owner_run_id='zero-available-original-owner')
    assert schedule['owner_run_id']=='zero-available-original-owner' and schedule['attempts']==[]
    assert len(schedule['pairs'])==34 and schedule['body_denominator']==34


def test_capability_refuses_unknown_fields_even_with_matching_bound_source_graph(sealed_metadata):
    m,body,configs,tmp=sealed_metadata
    plan=m.freeze_preparation_plan(body,configs,output=tmp/'prep-plan.json')
    issued=m.issue_preparation_capability(plan,output=tmp/'prep-cap.json');value=m.runtime.read_bound(issued['capability'])
    value['allow_test']=True
    changed=m.runtime.write_new(tmp/'unrecognized-capability.json',value)
    with pytest.raises(ValueError):m.verify_capability(changed,stage='preparation')


@pytest.mark.parametrize('fault',['none','config','prepared','source'])
def test_planning_entry_delegates_original_once_and_rebinds_exact_sources(tmp_path,monkeypatch,fault):
    import sys,json
    from sim.ausc_policy import geometry_plan,process_guard
    m=module();prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'scene.bin').write_bytes(b'synthetic')
    inventory=m._tree(prepared);source=m.runtime.write_new(tmp_path/'geometry.usdc',{'synthetic_fixture_only':True})
    config={'geometry_source':source['path'],'geometry_source_sha256':source['sha256'],
        m.PLANNING_KEY:dict(prepared_inventory=dict(path='/synthetic/inventory.json',sha256='0'*64))}
    cfg=m.runtime.write_new(tmp_path/'config.json',config);output=tmp_path/'output';calls=[]
    monkeypatch.setattr(m,'planning_preflight',lambda c,**kw:dict(prepared=dict(output=str(prepared),inventory=inventory)))
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(process_guard,'process_identity',lambda pid:dict(pid=pid,pgid=1,sid=1,start_ticks='1',state='R'))
    original_sources=m.planning_sources();counter=[]
    def sources():
        counter.append(1)
        return dict(original_sources,mutation={}) if fault=='source' and len(counter)>1 else original_sources
    monkeypatch.setattr(m,'planning_sources',sources)
    def delegate():
        calls.append(list(sys.argv));output.mkdir();m.runtime.write_new(output/'COMPLETE.json',{'completed':True})
        if fault=='config':Path(cfg['path']).write_text(json.dumps(dict(config,changed=True)))
        elif fault=='prepared':(prepared/'scene.bin').write_bytes(b'changed')
    monkeypatch.setattr(geometry_plan,'main',delegate)
    if fault=='none':
        receipt=m.run_planning(cfg,output)
        assert m.runtime.read_bound(receipt)['configuration']==cfg and len(calls)==1
        assert calls[0][-4:]==['--config',cfg['path'],'--output',str(output)]
    else:
        with pytest.raises(ValueError):m.run_planning(cfg,output)
        assert not (output/'heldout_planning.json').exists() and len(calls)==1


def test_explicit_heldout_expert_wraps_original_probe_once_and_preserves_failed_arm(tmp_path,monkeypatch):
    from sim.ausc_policy import fresh_expert
    m=module();cap=dict(model={'checkpoint':{'sha256':'a'*64}});row=dict(episode_id='test-expert',subject_id='p000000',arm='expert',seed=1)
    reference=dict(path='/synthetic/native-cap.json',sha256='b'*64)
    config={m.ROUTE_KEY:dict(schema=m.NATIVE_ROUTE_SCHEMA,capability=reference,episode_id=row['episode_id'],arm='expert')}
    binding=dict(stage='native',capability=cap,reference=reference,row=row);calls=[]
    monkeypatch.setattr(m,'preflight',lambda *a,**k:binding)
    def original(*args,**kwargs):calls.append((args,kwargs));raise RuntimeError('original failed expert setup')
    monkeypatch.setattr(fresh_expert,'_run_probe',original)
    result=fresh_expert.run_probe(None,None,tmp_path,config,{})
    assert len(calls)==1 and result['success'] is False and result['physical_validation'] is False
    assert result['heldout_evaluation']['arm']=='expert' and result['training_eligible'] is False
    assert 'original failed expert setup' in result['failure_reason'] and result['physics_steps']==0


def execution_header_fixture(tmp_path,monkeypatch):
    # Synthetic terminal/launch protocol, never a real heldout capability.
    from sim.ausc_policy import process_guard
    m=module();root=tmp_path/'runs/ausc_policy/fixed-pairs';out=root/'output';out.mkdir(parents=True)
    rows=[];jobs=[]
    for arm in m.ARMS:
        job='p000000-'+arm;cfg=m.runtime.write_new(tmp_path/(job+'.json'),{m.ROUTE_KEY:{'arm':arm}})
        rows.append(dict(episode_id=job,subject_id='p000000',arm=arm,owner_run_id=root.name,job_id=job,
            config=cfg['path'],output=str(out/job)))
        jobs.append(dict(id=job,config=cfg['path'],sha256=cfg['sha256']))
    cfg=m.runtime.write_new(tmp_path/'owner.json',dict(experiment='isaac_batch',base=str(tmp_path),parallelism=2,jobs=jobs))
    source=m.runtime.reference(process_guard.__file__)
    run=dict(run_id=root.name,code_commit='a'*40,config=m.runtime.read_bound(cfg),config_sha256=cfg['sha256'],
        command=[m.runtime.sys.executable,'-u','-B','-m','sim.ausc_policy.isaac_batch','--config',cfg['path'],'--output',str(out)],
        resource_guard=dict(path=source['path'],sha256=source['sha256'],gpu_exit_grace_s=process_guard.EXIT_GPU_GRACE_S),
        started_utc='2026-10-27T00:00:00Z',completed=False)
    m.runtime.write_new(root/'run.json',run)
    stage=dict(passed=False,exit_code=-2,process_group_drained=True,errors=['original resource abort'],signals=['SIGINT'],
        deferred_signals=[],process_identity=dict(pid=100,pgid=100,sid=100,start_ticks='1',members={'100':'1','101':'2'}))
    terminal=m.runtime.write_new(root/'result.json',dict(run,stage=stage,gpu_released=True,ended_utc='2026-10-27T00:01:00Z'))
    processing=dict(processing_freeze=m.runtime.write_new(tmp_path/'processing.json',dict(reviewed_commit='a'*40)))
    cap=dict(attempts=rows,processing=processing,pair_plan={'path':'/synthetic/pair.json','sha256':'b'*64})
    monkeypatch.setattr(m,'verify_pair_plan',lambda *a,**kw:dict(schedule={'owner_run_id':root.name}))
    monkeypatch.setattr(m,'preflight',lambda c,**kw:dict(row=next(r for r in rows if r['arm']==c[m.ROUTE_KEY]['arm']),
        reference={'path':'/synthetic/cap.json','sha256':'c'*64}))
    return m,cap,terminal,root,jobs


@pytest.mark.parametrize('fault',['none','result_without_launch','unlisted_launch','identity','config','not_drained'])
def test_terminal_inventory_uses_actual_launch_and_preserves_missing_result_and_unlaunched(tmp_path,monkeypatch,fault):
    m,cap,terminal,root,jobs=execution_header_fixture(tmp_path,monkeypatch);out=root/'output'
    identity=dict(pid=101,pgid=100,sid=100,start_ticks='2',state='S')
    launch=dict(id=jobs[0]['id'],config_sha256=jobs[0]['sha256'],pid=101,identity=identity,output=str(out/jobs[0]['id']))
    if fault=='identity':launch['identity']['start_ticks']='recycled'
    if fault=='config':launch['config_sha256']='0'*64
    if fault=='not_drained':
        value=m.runtime.read_bound(terminal);value['stage']['process_group_drained']=False
        Path(terminal['path']).unlink();terminal=m.runtime.write_new(terminal['path'],value)
    if fault=='result_without_launch':m.runtime.write_new(out/(jobs[0]['id']+'.result.json'),{})
    else:m.runtime.write_new(out/((jobs[0]['id'] if fault!='unlisted_launch' else 'unlisted')+'.launch.json'),launch)
    if fault=='none':
        rows=m._execution_inventory(cap,terminal,{'path':'/synthetic/cap.json','sha256':'c'*64})['attempts']
        assert rows[0]['launched'] is True and rows[0]['child_result'] is None and rows[0]['complete'] is None
        assert rows[1]['launched'] is False and rows[1]['child_result'] is None
    else:
        with pytest.raises(ValueError):m._execution_inventory(cap,terminal,{'path':'/synthetic/cap.json','sha256':'c'*64})


@pytest.mark.parametrize('fault',['none','owner_commit','owner_command'])
def test_native_owner_rebinds_original_command_and_reviewed_processing_commit(tmp_path,monkeypatch,fault):
    m,cap,terminal,root,jobs=execution_header_fixture(tmp_path,monkeypatch);out=root/'output';row=cap['attempts'][0]
    reference={'path':'/synthetic/cap.json','sha256':'c'*64}
    monkeypatch.setattr(m,'preflight',lambda *a,**kw:dict(reference=reference,row=row,capability=cap))
    value=m.runtime.read_bound(terminal);value['completed']=True
    value['stage'].update(passed=True,exit_code=0,errors=[],signals=[])
    if fault=='owner_commit':value['code_commit']='b'*40
    if fault=='owner_command':value['command'][4]='sim.ausc_policy.unapproved'
    Path(terminal['path']).unlink();terminal=m.runtime.write_new(terminal['path'],value)
    run={k:v for k,v in value.items() if k not in ('stage','gpu_released','ended_utc')};run['completed']=False
    (root/'run.json').unlink();m.runtime.write_new(root/'run.json',run)
    report=m.runtime.write_new(Path(row['output'])/'COMPLETE.json',dict(runtime_shutdown=dict(app_close_returned=True,
        fast_shutdown=False,interpreter_exit_policy='success_only_os_exit_after_native_close_and_fsynced_receipt')))
    identity=dict(pid=101,pgid=100,sid=100,start_ticks='2',state='S')
    launch=dict(id=row['job_id'],config_sha256=jobs[0]['sha256'],pid=101,identity=identity,output=row['output'])
    m.runtime.write_new(out/(row['job_id']+'.launch.json'),launch)
    m.runtime.write_new(out/(row['job_id']+'.result.json'),dict(launch,exit_code=0,worker_complete=True,physical_pass=False,result_sha256=report['sha256']))
    if fault=='none':assert m._native_owner(row,terminal,reference)['complete']==report
    else:
        with pytest.raises(ValueError):m._native_owner(row,terminal,reference)


@pytest.mark.parametrize('owner_normal',[True,False])
def test_arm_classification_never_promotes_physics_under_failed_owner_and_preserves_reason_flags(owner_normal):
    m=module();proof=dict(launched=True,complete={'sha256':'a'*64},child_normal=True,actual_physics_steps=12,
        physical_passed=True,source_verified=True,trace_verified=True,act_verified=True,video_verified=True,audit_errors=[])
    value=m._arm_status(proof,owner_normal=owner_normal)
    assert value['status']==('passed' if owner_normal else 'infrastructure_abort')
    proof.update(physical_passed=False,child_normal=False)
    value=m._arm_status(proof,owner_normal=owner_normal)
    assert value['status']=='infrastructure_abort' and value['reason_flags']['physical_failure'] is True
    assert value['reason_flags']['lifecycle_failure'] is True
    proof.update(complete=None)
    assert m._arm_status(proof,owner_normal=owner_normal)['status']=='missing_result'
    proof.update(launched=False,actual_physics_steps=0)
    assert m._arm_status(proof,owner_normal=owner_normal)['status']=='planned_unlaunched'


def test_measured_summary_preserves_raw_dynamics_force_hold_and_actual_owner_counts(tmp_path):
    m,events,ipc,run,dep,permit=act_audit_fixture(tmp_path)
    for index,event in enumerate(events):
        event['measurements']=dict(hall_N=4. if index<12 else 0.,compression_m=.0025,diaphragm_N=4.,retainer_N=0.)
        event['native_row']=dict(maximum_velocity_feedforward=0.)
    run['runtime_cost']={'probe_wall_s':3.};run['l2_handoffs']=[dict(l2_coarse_reached=False,l3_completed=False)]
    result=m._measured_metrics(run,events)
    assert result['actual_physics_steps']==24 and result['actual_simulation_s']==.1
    assert result['actual_owner_ticks']=={'ACT_L2':24} and result['measurement_extrema']['hall_N']==dict(min=0.,max=4.)
    assert result['command_fd']['max_velocity_rad_s']==result['observed_fd']['max_velocity_rad_s']==0.
    assert result['reported_l2_handoffs']==run['l2_handoffs'] and result['reported_runtime_cost']==run['runtime_cost']


@pytest.mark.parametrize('fault',['none','command_speed','command_history','contact_acceleration'])
def test_independent_command_replay_preserves_original_whole_stream_gates(tmp_path,fault):
    m,events,*_=act_audit_fixture(tmp_path)
    for event in events:event['controller_before'].update(previous_command_q=[0.]*7,contact_active=False)
    if fault=='command_speed':events[-1]['q_command'][0]=.001
    elif fault=='command_history':events[-1]['controller_before']['previous_command_q'][0]=.01
    elif fault=='contact_acceleration':
        for i,event in enumerate(events):
            event['q_command']=[.3*(i/240.)**2]+[0.]*6
            event['controller_before'].update(previous_command_q=events[i-1]['q_command'] if i else [0.]*7,contact_active=True)
    if fault=='none':assert m._command_gates(events)['checked_ticks']==24
    else:
        with pytest.raises(ValueError):m._command_gates(events)


def test_actual_provider_envelope_is_bound_before_first_motion_and_rejects_late_binding(tmp_path):
    from test_ausc_given_contact_pose import reference,metadata
    from sim.ausc_policy import m2_given_contact_pose as pose
    m=module();ref=reference(tmp_path);meta=metadata(ref)
    provider=pose.GivenPoseProvider(ref,'synthetic-episode',meta)
    provider.bind_before_motion(SimpleNamespace(env=SimpleNamespace(rows=[],base=np.asarray(ref.snapshot()['world_from_base']),
        state=lambda:dict(time_s=0.))))
    assert m._reference_availability(provider.snapshot(),meta) is True
    late=provider.snapshot();late['availability_binding']['physics_row']=1
    with pytest.raises(ValueError):m._reference_availability(late,meta)


def test_arm_audit_reads_original_fresh_expert_inputs_before_motion_receipt(tmp_path,monkeypatch):
    from test_ausc_given_contact_pose import reference,metadata
    from sim.ausc_policy import m2_given_contact_pose as pose
    m=module();ref=reference(tmp_path);meta=metadata(ref);provider=pose.GivenPoseProvider(ref,'synthetic-episode',meta)
    provider.bind_before_motion(SimpleNamespace(env=SimpleNamespace(rows=[],base=np.asarray(ref.snapshot()['world_from_base']),
        state=lambda:dict(time_s=0.))))
    row=dict(output=str(tmp_path),episode_id='synthetic-episode',job_id='synthetic-job',subject_id='synthetic-id',arm='expert',metadata=meta)
    cap_ref={'path':'/synthetic/cap.json','sha256':'a'*64};cap=dict(model={'synthetic':True},source_inventory={})
    tag=dict(capability=cap_ref,episode_id=row['episode_id'],subject_id=row['subject_id'],arm='expert',
        capability_authenticated=True,model=cap['model'],protocol=m.evaluation_protocol(),simulation_only=True)
    run=dict(heldout_evaluation=tag,runtime_sources_sha256={},given_reference=provider.snapshot(),
        executed_controller_class='sim.ausc_policy.m2_given_contact_controller.GivenContactExpert',learned_policy=False,
        handoff_policy_id=m.runtime.handoff.POLICY_ID,action_semantics_id=m.runtime.given.ACTION_SEMANTICS_ID)
    complete=m.runtime.write_new(tmp_path/'run-result.json',run)
    config=m.runtime.write_new(tmp_path/'config.json',{})
    snapshot=tmp_path/'fresh_expert_inputs/given_reference_before_motion.json'
    m.runtime.write_new(snapshot,run['given_reference'])
    header=dict(attempt=row,launched=True,complete=complete,config=config,child_result=None,identity=None)
    monkeypatch.setattr(m,'_read_native_events',lambda *a:None)
    _,proof=m._inspect_arm(header,cap,cap_ref,{},{});assert proof['source_verified'] is True
    # A same-content alias cannot replace the actual native writer's role.
    m.runtime.write_new(tmp_path/'snapshot/given_reference_before_motion.json',run['given_reference'])
    snapshot.unlink()
    _,proof=m._inspect_arm(header,cap,cap_ref,{},{});assert proof['source_verified'] is False


def test_storage_union_charges_distinct_paths_and_never_uses_planned_pair_denominator(tmp_path):
    m=module();first=tmp_path/'first';first.mkdir();(first/'a').write_bytes(b'same bytes')
    second=tmp_path/'second';second.mkdir();(second/'b').write_bytes(b'same bytes')
    trees=[m._tree(first),m._tree(second)];ref=m.runtime.reference(first/'a')
    result=m._storage_union(trees,[ref],observed_jobs=0)
    assert result['charged_path_bytes']==20 and result['charged_files']==2 and result['physical_inode_bytes']==20
    assert result['bytes_per_observed_job'] is None and result['observed_jobs']==0
    result=m._storage_union(trees,[ref],observed_jobs=2)
    assert result['bytes_per_observed_job']==10.
    (first/'a').write_bytes(b'changed!!!')
    with pytest.raises(ValueError):m._storage_union(trees,[ref],observed_jobs=2)


@pytest.mark.parametrize('failure_receipt',[True,False])
def test_summary_storage_preserves_actual_failed_job_cost_without_claiming_complete_shared_union(tmp_path,monkeypatch,failure_receipt):
    from sim.ausc_policy import scene_store
    m=module();root=tmp_path/'campaign';root.mkdir();batch=tmp_path/'native-owner';job=batch/'output'/'failed';job.mkdir(parents=True)
    if failure_receipt:(job/'FAILED.json').write_text('{"error":"before scene publication"}')
    else:(job/'launch.json').write_text('{"launched":true}')
    config=m.runtime.write_new(root/'config.json',dict(compact_scene_storage=dict(sources={})))
    empty=m.runtime.write_new(root/'empty.json',{})
    smokes=m.runtime.write_new(root/'smokes.json',dict(owner_inventories=[],permit=empty))
    body=dict(layout=dict(root=str(root),planning_owner=dict(output=str(tmp_path/'planning-owner'/'output')),
        preparation_owner=dict(output=str(tmp_path/'preparation-owner'/'output'))),bridge_smokes=smokes)
    for name in ('planning-owner','preparation-owner'):(tmp_path/name).mkdir()
    cap=dict(prepared_inventory=empty,processing=dict(processing_freeze=empty))
    execution=dict(root=str(batch),attempts=[dict(attempt=dict(output=str(job),job_id='failed'),launched=True,config=config)])
    bundle=dict(admission={},complete={},software={},config=dict(dataset={}),owner={})
    monkeypatch.setattr(m.deployment,'verify_permit',lambda *a:dict(attempts=[]))
    monkeypatch.setattr(scene_store,'_sdk_bundle',lambda *a:None)
    result=m._summary_storage(empty,cap,body,{},execution,bundle,output=tmp_path/'summary.json')
    assert result['observed_jobs']==1 and result['charged_path_bytes']>=result['whole_jobs'][0]['total_bytes']
    assert result['whole_jobs'][0]['total_bytes']>0
    assert result['input_cost_scope_complete'] is False and result['original_native_shared_budget_passed'] is False
    if failure_receipt:
        assert result['original_native_batch_ledger']['attempts_without_scene_reference']==[str(job)]
        assert result['original_native_batch_ledger_error'] is None
    else:assert 'scene_ref.json and FAILED.json' in result['original_native_batch_ledger_error']


def test_unissued_summary_refuses_before_any_native_or_test_payload(tmp_path,monkeypatch):
    m=module();ref=m.runtime.write_new(tmp_path/'unissued.json',dict(schema=m.CAPABILITY_SCHEMA))
    monkeypatch.setattr(m,'_read_native_events',lambda *a:pytest.fail('native trace before authorization'))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('body before authorization'))
    with pytest.raises(ValueError):m.summarize(ref,None,output=tmp_path/'summary.json')
    assert not (tmp_path/'summary.json').exists()


def test_aborted_cpu_still_exposes_bound_returned_chunks_but_cannot_qualify_normal_exit(tmp_path,monkeypatch):
    import json
    m=module();parent=dict(pid=10,pgid=10,sid=10,start_ticks='1',state='S')
    child=dict(pid=11,pgid=10,sid=10,start_ticks='2',state='R')
    dep=dict(metadata={'synthetic':True},permit={'path':'/synthetic/cap.json','sha256':'a'*64})
    permit=dict(model={'fixture_only':True},admission={'path':'/synthetic/admission.json','sha256':'b'*64})
    ready=dict(identity=child,parent_identity=parent,execution={'synthetic_only':True},bound='exact')
    records=[dict(event='ready',value=ready),dict(event='inference',response={'entire_h20':'retained'},accepted=False)]
    (tmp_path/'given_act_ipc.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (tmp_path/'given_act_cpu_stderr.log').write_text('original timeout')
    exit=dict(schema='m2_given_contact_cpu_child_exit_v1',identity=child,parent_identity=parent,
        started=True,reaped=True,normal_exit=False,exit_code=-15,signals=['SIGTERM'],errors=[],shared_native_group_signaled=False,
        request_count=1,trace=m.runtime.reference(tmp_path/'given_act_ipc.jsonl'),stderr=m.runtime.reference(tmp_path/'given_act_cpu_stderr.log'))
    m.runtime.write_new(tmp_path/'given_act_cpu_exit.json',exit)
    run=dict(cpu_child_exit=exit,learned_policy_receipt=dict(model=permit['model'],admission=permit['admission'],permit=dep['permit'],
        metadata=dep['metadata'],child_runtime=ready,expert_l2_fallback=False))
    monkeypatch.setattr(m.deployment,'_ready_binding',lambda *a:dict(bound='exact'))
    monkeypatch.setattr(m.runtime,'validate_execution_settings',lambda *a,**kw:None)
    observed,evidence=m._cpu_exchange_records(tmp_path,run,dep,permit,parent,{'admission':{'source_inventory':{}},'software':{}})
    assert observed==records and evidence['normal_exit'] is False
    with pytest.raises(ValueError):m._cpu_evidence(tmp_path,run,dep,permit,parent,{'admission':{'source_inventory':{}},'software':{}})


def test_terminal_summary_retains_missing_and_unlaunched_mates_under_original34(tmp_path,monkeypatch):
    # Only metadata arithmetic/dispatch is tested: actual authority is mocked,
    # and no result here may satisfy a real native or heldout qualification.
    m,layout=metadata_layout();inventory=m.planning_inventory(layout,planning_records(layout,found=1))
    schedule=m.paired_schedule(layout,inventory,owner_run_id='synthetic-only');rows=schedule['attempts']
    assert len(rows)==2
    cap=dict(pair_plan=m.runtime.write_new(tmp_path/'pair.json',dict(schedule=schedule)),body_plan=m.runtime.write_new(tmp_path/'body.json',dict(layout=layout)),
        admission={'path':'/synthetic/admission.json','sha256':'a'*64},model={'fixture_only':True},processing={},source_inventory={})
    ref=m.runtime.write_new(tmp_path/'cap.json',cap)
    headers=[dict(attempt=row,config={},launched=i==0,launch=None,child_result=None,complete=None,identity=None) for i,row in enumerate(rows)]
    execution=dict(owner_terminal=None,owner_normal=False,attempts=headers,root=None)
    monkeypatch.setattr(m,'verify_capability',lambda *a,**kw:deepcopy(cap))
    monkeypatch.setattr(m,'verify_pair_plan',lambda *a,**kw:dict(schedule=deepcopy(schedule),planning_inventory=inventory))
    monkeypatch.setattr(m,'verify_body_plan',lambda *a:dict(layout=layout))
    monkeypatch.setattr(m,'_execution_inventory',lambda *a:deepcopy(execution))
    monkeypatch.setattr(m.deployment,'_admission_metadata',lambda *a:{'software':{}})
    monkeypatch.setattr(m.runtime,'software_inventory',lambda *a:{})
    monkeypatch.setattr(m,'_summary_storage',lambda *a,**kw:dict(input_cost_scope_complete=False,observed_jobs=1))
    result=m.runtime.read_bound(m.summarize(ref,None,output=tmp_path/'summary.json'))
    assert result['counts']['body_denominator']==34 and result['counts']['available_bodies']==1
    assert [r['status'] for r in result['outcomes']]==['missing_result','planned_unlaunched']
    assert result['counts']['executed_pairs']==0 and result['heldout_evaluation_complete'] is False
    assert len(result['review_videos'])==4 and result['milestone_reached'] is False
