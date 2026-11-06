"""Current-contract recovery supervision: CPU semantics, never native acceptance."""
from copy import deepcopy
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_ausc_given_contact_capture import new_inputs
from test_ausc_given_contact_handoff import FixtureTrajectory


def module():
    name='sim.ausc_policy.m2_given_contact_recovery'
    assert importlib.util.find_spec(name) is not None, 'Current-contract recovery route missing'
    return importlib.import_module(name)


def exporter():
    name='sim.ausc_policy.m2_given_contact_recovery_dataset'
    assert importlib.util.find_spec(name) is not None, 'Recovery label exporter missing'
    return importlib.import_module(name)


def source(tmp_path):
    from sim.ausc_policy import m2_given_contact_runtime as r
    ref,_,_,meta=new_inputs(tmp_path)
    meta['handoff_policy_id']=r.handoff.POLICY_ID
    cfg=dict(experiment='geometry_qualification',base='/workspace/wam',isaac_python='/opt/wam-runtime/envs/isaacsim60/bin/python',
        geometry_id=r.geometry.GEOMETRY_ID,geometry_source_sha256=r.geometry.GEOMETRY_SOURCE_SHA256,
        physics=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02),
        m2_given_contact_control=dict(schema=r.pose.TASK_ID,metadata=meta,episode_id=meta['episode_id'],handoff_policy_id=r.handoff.POLICY_ID),
        fresh_expert=dict(plans=[ref.snapshot()['sources']['fresh_plan']],passive_preload_N=1.5,
            raw_multimodal_capture=dict(enabled=True,profile='compact',episode_id=meta['episode_id'],compact=dict(metadata=deepcopy(meta)))),
        compact_scene_storage=dict(batch_id='original'))
    return r.reference(r.write_new(tmp_path/'original.json',cfg)['path'])


def test_derive_is_exact_runtime_projection_and_original_is_unchanged(tmp_path):
    m=module();s=source(tmp_path);before=open(s['path'],'rb').read()
    cfg=m.derive_config(s,episode_id='recovery-a015',seed=17,amplitude_rad=.015)
    got=m.preflight(cfg)
    assert got['metadata']['episode_id']=='recovery-a015'
    assert got['settings']['amplitude_rad']==.015
    assert cfg['physics']==dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02)
    assert open(s['path'],'rb').read()==before
    cfg['fresh_expert']['passive_preload_N']=2.
    with pytest.raises(ValueError,match='projection'):m.preflight(cfg)


@pytest.mark.parametrize('fault',['validation','test','physics','physics_type','native','geometry','extra','amplitude','seed','mixed'])
def test_preflight_rejects_changed_scope_physics_or_runtime(tmp_path,fault):
    m=module();s=source(tmp_path);cfg=m.derive_config(s,episode_id='recovery-a030',seed=17,amplitude_rad=.03)
    if fault in ('validation','test'):cfg['m2_given_contact_control']['metadata']['split']=fault
    elif fault=='physics':cfg['physics']['position_iterations']=32
    elif fault=='physics_type':cfg['physics']['velocity_iterations']=True
    elif fault=='native':cfg['isaac_python']='other-python'
    elif fault=='geometry':cfg['geometry_id']='old-tool'
    elif fault=='extra':cfg[m.ROUTE_KEY]['unexpected']=True
    elif fault=='amplitude':cfg[m.ROUTE_KEY]['amplitude_rad']=.3
    elif fault=='seed':cfg[m.ROUTE_KEY]['seed']=True
    else:cfg['m2_given_contact_pilot_deployment']={}
    with pytest.raises(ValueError):m.preflight(cfg)


def trajectory():
    from sim.ausc_policy import m2_given_contact_handoff as h
    return FixtureTrajectory(np.zeros(7),dt=h.DT,**h.PLANNING_LIMITS)


def test_preview_retains_nonzero_command_acceleration_and_never_touches_live_state():
    m=module();from sim.ausc_policy import m2_given_contact_handoff as h
    t=trajectory();t.input.current_acceleration=[1e-8]*7;t.acceleration=np.full(7,1e-8)
    before=h.input_snapshot(t);objects=(t,t.input,t.output,t.generator);checked=[]
    preview=m.preview_cells(t,np.zeros((3,7)),[np.zeros(7)]*2,lambda q:checked.append(q.copy()),
        trajectory_factory=FixtureTrajectory)
    assert h.input_snapshot(t)==before
    assert preview['input_state']['fields']['current_acceleration']==[1e-8]*7
    assert all(a is b for a,b in zip(objects,(t,t.input,t.output,t.generator)))
    assert len(checked)==24 and len(preview['commands'])==24


def test_preview_rejects_collision_and_a_disconnected_command_seam():
    m=module();t=trajectory()
    with pytest.raises(ValueError,match='blocked'):
        m.preview_cells(t,np.zeros((3,7)),[np.zeros(7)],lambda q:(_ for _ in ()).throw(ValueError('blocked')),
            trajectory_factory=FixtureTrajectory)
    bad=np.zeros((3,7));bad[-1,0]=.01
    with pytest.raises(ValueError,match='seam'):
        m.preview_cells(t,bad,[np.zeros(7)],lambda q:None,trajectory_factory=FixtureTrajectory)


def query(tmp_path):
    from sim.ausc_policy import m2_contract as c
    from sim.ausc_policy import m2_given_contact_pose as pose
    from test_ausc_given_contact_pose import bound
    ref,_,_,meta=new_inputs(tmp_path)
    value=ref.snapshot();plan=pose.bound_json(value['sources']['fresh_plan'])
    plan['transfer_q']=[np.zeros(7).tolist(),[.02]*7,np.zeros(7).tolist()]
    value['sources']['fresh_plan']=bound(tmp_path,'query-plan.json',plan)
    ref=pose.GivenReference.from_snapshot(value);meta=pose.route_metadata(meta,ref)
    q=np.zeros(7);state=c.encode_state(q,ref.pose_base,1.,metadata=meta)
    return ref,dict(measured_q=q.tolist(),measured_dq=[0.]*7,history=[state.tolist()]*2,
        history_times=[.95,1.],time_s=1.,physics_row=240,reference_sha256=ref.sha256,
        metadata=meta,context=[meta['episode_id'],meta['target_id'],0,'APPROACH'],hall_N=0.,physical_contact_N=0.,compression_m=0.)


def test_same_state_teacher_replans_from_measured_and_retained_state(tmp_path):
    m=module();ref,q=query(tmp_path);t=trajectory();calls=[]
    q['measured_q'][0]=.001
    from sim.ausc_policy import m2_contract as c
    q['history'][-1]=c.encode_state(q['measured_q'],ref.pose_base,1.,metadata=q['metadata']).tolist()
    def planner(start,goal,deadline):
        calls.append((start.copy(),goal.copy()));return [start,goal],{'reason':'checked'}
    out=m.propose_recovery(q,t,np.zeros((3,7)),np.zeros(7),reference=ref,planner=planner,
        verify_command=lambda q:None,trajectory_factory=FixtureTrajectory)
    # The intentionally nonmoving fixture cannot reach the connector. It must
    # abstain rather than claim its measured start is an executed trajectory.
    assert out['kind']=='abstain' and 'bound' in out['reason']
    assert calls[0][0][0]==.001
    assert out['query']['measured_q'][0]==.001
    assert out['physics_executed'] is False and out['training_eligible'] is False


@pytest.mark.parametrize('fault',['contact','phase','revision','history','velocity','clock','reference','state17','goal'])
def test_same_state_teacher_abstains_on_invalid_or_unsupported_state(tmp_path,fault):
    m=module();ref,q=query(tmp_path)
    if fault=='contact':q['physical_contact_N']=.2
    elif fault=='phase':q['context'][-1]='FREE'
    elif fault=='revision':q['context'][2]=1
    elif fault=='history':q['history_times'][0]=.9
    elif fault=='velocity':q['measured_dq'][0]=.176
    elif fault=='clock':q['physics_row']=241
    elif fault=='reference':q['reference_sha256']='0'*64
    elif fault=='state17':q['history'][-1][7]+=.01
    goal=np.ones(7) if fault=='goal' else np.zeros(7)
    called=[]
    out=m.propose_recovery(q,trajectory(),np.zeros((3,7)),goal,reference=ref,
        planner=lambda *a:called.append(a),verify_command=lambda q:None,trajectory_factory=FixtureTrajectory)
    assert out['kind']=='abstain' and not called


def ledger(audit,meta):
    m=module();events=[]
    for event in audit:
        raw=event.get('raw_intent')
        if raw is None or raw['native_tick_in_cell']!=0:continue
        s=raw['setter'];events.append(dict(intent_id=s['intent_id'],setter_sha256=s['setter_sha256'],
            physics_row_start=s['physics_row_start'],physics_row_end_exclusive=s['physics_row_start']+12,
            kind='recovery_teacher',completed=True))
    for event in events[:10]:event['kind']='perturbation'
    return dict(schema=m.SIDECAR_SCHEMA,episode_id=meta['episode_id'],reference_sha256=meta['reference_sha256'],
        config_sha256='a'*64,completed=True,native_steps=len(audit),events=events,
        perturbations_executed=1,queries=[dict(kind='recovery_proposal')])


def test_labels_are_raw_intents_and_perturbation_can_only_be_in_causal_history(tmp_path):
    m=exporter();ref,rows,audit,meta=new_inputs(tmp_path,n=34);sidecar=ledger(audit,meta)
    got=m.assemble_recovery_samples(rows,audit,meta,sidecar)
    assert len(got['states'])==5
    assert got['actions'][0,0,0]==pytest.approx(.720-.109)
    assert got['derived_index']['windows'][0]['inclusive_native_ticks'][0]==119
    sidecar['events'][10]['kind']='excluded'
    got=m.assemble_recovery_samples(rows,audit,meta,sidecar)
    assert len(got['states'])==4
    assert got['derived_index']['windows'][0]['inclusive_native_ticks'][0]==131


@pytest.mark.parametrize('fault',['unknown','missing','hash','interval','failed','nominal'])
def test_missing_false_or_non_teacher_provenance_never_produces_positive_labels(tmp_path,fault):
    m=exporter();ref,rows,audit,meta=new_inputs(tmp_path,n=34);sidecar=ledger(audit,meta)
    if fault=='unknown':sidecar['events'][3]['kind']='guessed_teacher'
    elif fault=='missing':sidecar['events'].pop()
    elif fault=='hash':sidecar['events'][3]['setter_sha256']='f'*64
    elif fault=='interval':sidecar['events'][3]['physics_row_end_exclusive']-=1
    elif fault=='failed':sidecar['completed']=False
    else:
        for event in sidecar['events'][10:]:event['kind']='nominal_teacher'
    if fault=='nominal':assert len(m.assemble_recovery_samples(rows,audit,meta,sidecar)['states'])==0
    else:
        with pytest.raises(ValueError):m.assemble_recovery_samples(rows,audit,meta,sidecar)


def test_scene_projection_strips_only_validated_recovery_intent(tmp_path,monkeypatch):
    from test_ausc_given_contact_scene_store import given_config
    from sim.ausc_policy import scene_store,m2_given_contact_runtime as r
    m=module();cfg,_=given_config(tmp_path,monkeypatch)
    cfg['physics']=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02)
    cfg['isaac_python']=m.NATIVE_PYTHON
    cfg['m2_given_contact_control']['handoff_policy_id']=r.handoff.POLICY_ID
    cfg['m2_given_contact_control']['metadata']['handoff_policy_id']=r.handoff.POLICY_ID
    cfg['fresh_expert']['raw_multimodal_capture']['compact']['metadata']['handoff_policy_id']=r.handoff.POLICY_ID
    src=r.write_new(tmp_path/'source-config.json',cfg)
    derived=m.derive_config(src,episode_id='new-recovery',seed=17,amplitude_rad=.015)
    assert scene_store._authoring(derived)==scene_store._authoring(cfg)
    derived['fresh_expert']['passive_preload_N']=2.
    with pytest.raises(ValueError):scene_store._authoring(derived)


def test_real_ruckig_0153_moving_full_path_and_live_execution_parity(tmp_path):
    ruckig=pytest.importorskip('ruckig');assert ruckig.__version__=='0.15.3'
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    from sim.ausc_policy import m2_given_contact_handoff as h,m2_contract as c
    m=module();ref,q=query(tmp_path)
    t=JointTrajectory(np.zeros(7),dt=h.DT,**h.PLANNING_LIMITS);t.set_target(np.full(7,.02))
    commands=[np.zeros(7)]*3
    for _ in range(24):commands.append(t.step())
    assert np.max(abs(t.velocity))>0 and np.max(abs(t.acceleration))>0
    q['measured_q']=commands[-1].tolist();q['measured_dq']=t.velocity.tolist()
    q['history']=[c.encode_state(p,ref.pose_base,1.,metadata=q['metadata']).tolist()
        for p in (commands[-13],commands[-1])]
    before=h.input_snapshot(t);output=h.output_snapshot(t);objects=(t,t.input,t.output,t.generator)
    checked=[]
    out=m.propose_recovery(q,t,np.asarray(commands[-3:]),np.full(7,.02),reference=ref,
        planner=lambda start,goal,deadline:([start,goal],{'reason':'direct_checked_fixture'}),
        verify_command=lambda command:checked.append(command.copy()))
    assert out['kind']=='recovery_proposal',out.get('reason')
    assert h.input_snapshot(t)==before and h.output_snapshot(t)==output
    actual=[]
    for target in out['preview']['cells']:
        t.set_target(np.asarray(target))
        for _ in range(12):actual.append(t.step())
    error=float(np.max(abs(np.asarray(actual)-np.asarray(out['preview']['commands']))))
    # A fresh preview generator recomputes from the same q/v/a; the live
    # generator retains its earlier polynomial. Preserve that tiny FP delta.
    np.testing.assert_allclose(actual,out['preview']['commands'],atol=1e-12,rtol=0)
    assert len(checked)>=len(actual) and all(a is b for a,b in zip(objects,(t,t.input,t.output,t.generator)))
    assert np.max(abs(t.velocity))<=1e-10 and np.max(abs(t.acceleration))<=1e-10
    receipt=dict(ruckig_version=ruckig.__version__,initial_input=before,steps=len(actual),
        preview_commands_sha256=out['preview']['commands_sha256'],preview_live_max_abs_rad=error,
        preview_live_tolerance_rad=1e-12,
        nonzero_initial_command_velocity=True,nonzero_initial_command_acceleration=True,
        native_physics_executed=False,training_eligible=False)
    # Optional evidence destination is an explicitly supplied local runtime path.
    import os
    if os.environ.get('WAM_RECOVERY_CPU_RECEIPT'):
        from sim.ausc_policy import m2_given_contact_runtime as r
        receipt['source_module']=r.reference(m.__file__)
        receipt['test_module']=r.reference(__file__)
        r.write_new(os.environ['WAM_RECOVERY_CPU_RECEIPT'],receipt)


def test_controller_records_executed_setter_intervals_without_relabeling_raw_targets():
    m=module();c=m.GivenContactRecoveryExpert.__new__(m.GivenContactRecoveryExpert)
    c._recovery_events=[];c._data_kind='perturbation';c.control_owner='TEACHER_L2'
    c.reference=SimpleNamespace(sha256='a'*64);c.trajectory=trajectory();c._intent_dispatcher=None
    c.env=SimpleNamespace(rows=[None]*240,state=lambda:dict(time_s=len(c.env.rows)/240.))
    c._issue_intent(np.full(7,.03),['e','L1',0,'APPROACH'],'given_l2_motion')
    assert c._recovery_events[0]['kind']=='perturbation'
    setter=c._dispatcher().events[0]
    assert setter['target_q']==[.03]*7 and c._recovery_events[0]['setter_sha256']==setter['setter_sha256']
    assert c._recovery_events[0]['completed'] is False
    assert c.trajectory.input.current_position==[0.]*7  # Issuing is not executing.


def test_unknown_and_failed_native_outcomes_never_contribute_export_arrays(tmp_path,monkeypatch):
    m=exporter();from sim.ausc_policy import m2_given_contact_runtime as r
    rows=[dict(episode_id='failed',reset_id='reset-f',run_id='run',job_id='f',split='train',subject_id='p000439',
        capture_root=str(tmp_path/'f/compact_multimodal'),run_result=dict(path=str(tmp_path/'f/COMPLETE.json'),sha256='a'*64)),
        dict(episode_id='unknown',reset_id='reset-u',run_id='run',job_id='u',split='train',subject_id='p000439',
        capture_root=str(tmp_path/'u/compact_multimodal'),run_result=dict(path=str(tmp_path/'u/COMPLETE.json'),sha256='b'*64))]
    manifest=r.write_new(tmp_path/'manifest.json',dict(schema=m.MANIFEST_SCHEMA,episodes=rows))
    monkeypatch.setattr(m,'read_episode',lambda row,base:(_ for _ in ()).throw(ValueError('physical failure or missing receipt')))
    result=r.read_bound(m.export_dataset(manifest,tmp_path/'out'))
    assert result['attempted_cases']==2 and result['positive_cases']==0 and result['windows']==0
    assert all(row['positive_data'] is False for row in result['outcomes'])
    assert not (tmp_path/'out/train.npz').exists() and result['training_eligible'] is False


def test_sidecar_ends_last_stop_at_actual_raw_evidence_not_later_align_ticks(tmp_path):
    m=exporter();ref,rows,audit,meta=new_inputs(tmp_path,n=34);sidecar=ledger(audit,meta)
    # Later L3/ALIGN ticks contain no raw setter. Their actual audit is retained
    # but cannot extend the preceding setter's owned interval or label cells.
    end=len(audit);audit.extend(dict(physics_row=end+i,raw_intent=None) for i in range(12))
    sidecar['native_steps']=len(audit)
    roles=m.validate_sidecar(sidecar,audit,meta)
    assert roles[end-12]['physics_row_end_exclusive']==end


@pytest.mark.parametrize('amplitude',[.015,.03])
def test_real_ruckig_dart_controller_executes_noise_then_replans_without_reset(tmp_path,amplitude):
    """Synthetic measured feedback: no FCL, contacts, or native acceptance."""
    ruckig=pytest.importorskip('ruckig');assert ruckig.__version__=='0.15.3'
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    from sim.ausc_policy import m2_given_contact_handoff as h,m2_stage,m2_contract as contract
    m=module();ref,initial=query(tmp_path);meta=initial['metadata']
    c=m.GivenContactRecoveryExpert.__new__(m.GivenContactRecoveryExpert)
    c.reference=ref;c.metadata=meta;c.episode_id=meta['episode_id'];c.control_owner='TEACHER_L2'
    c._recovery_events=[];c._recovery_queries=[];c._data_kind='nominal_teacher'
    c._attempted=False;c._perturbations=0;c._recovery_completed=False;c._intent_dispatcher=None
    c.recovery_binding=dict(settings=dict(seed=17,amplitude_rad=amplitude))
    c.trajectory=JointTrajectory(np.zeros(7),dt=h.DT,**h.PLANNING_LIMITS)
    c.trajectory.set_target(np.full(7,.02));live=c.trajectory
    c.q=np.zeros(7);c.env=SimpleNamespace(rows=[dict(q_command=[0.]*7)]*252,last_contacts=dict(diaphragm=0.,retainer=0.))
    c.env.state=lambda:dict(q=c.q.copy(),dq=c.trajectory.velocity.copy(),time_s=len(c.env.rows)/240.,hall_N=0.,compression_m=0.)
    c._history=m2_stage.CausalHistory(1.);context=(meta['episode_id'],meta['target_id'],0,'APPROACH')
    state=contract.encode_state(c.q,ref.pose_base,1.,metadata=meta)
    c._history.context=context;c._history.frames=[(1.,state.copy()),(1.05,state.copy())];c._history.last_time=1.05
    def observe():
        stamp=c.env.state()['time_s'];encoded=contract.encode_state(c.q,ref.pose_base,1.,metadata=meta)
        c._history.observe(encoded,stamp,context)
        return dict(context=context,phase='APPROACH',observed_at_s=stamp)
    c._reference_observation=observe;c._handoff_ready=lambda obs:False;c._deadline=lambda:None
    def block(commands,subphase,active=False):
        assert active is False and subphase=='given_l2_motion'
        assert c.trajectory is live
        for command in commands:
            c.q=np.asarray(command).copy();c.env.rows.append(dict(q_command=c.q.tolist()))
    c._block=block
    c._stationary=lambda:np.max(abs(c.trajectory.velocity))<=1e-10
    c._verify_recovery_command=lambda q:None
    planner_queries=[]
    def planner(start,goal,deadline):
        planner_queries.append((start.copy(),h.input_snapshot(c.trajectory)))
        return [start,goal],dict(reason='CPU_direct_fixture')
    c._plan_inter_point_bridge=planner
    assert c._recover_to_destination(np.full(7,.02)) is False
    assert c._perturbations==1 and c._recovery_completed and c.trajectory is live
    noise=[e for e in c._recovery_events if e['kind']=='perturbation']
    teacher=[e for e in c._recovery_events if e['kind']=='recovery_teacher']
    assert len(noise)==10 and all(e['completed'] for e in noise) and teacher
    assert noise[-1]['physics_row_end_exclusive']==teacher[0]['physics_row_start']
    assert len(planner_queries)==1
    assert np.max(np.abs(planner_queries[0][1]['fields']['current_velocity']))>0
    assert np.max(np.abs(planner_queries[0][1]['fields']['current_acceleration']))>0
    np.testing.assert_allclose(c.q,.02,atol=1e-10,rtol=0)


def test_export_never_counts_one_native_process_as_two_episode_resets(tmp_path,monkeypatch):
    m=exporter();from sim.ausc_policy import m2_given_contact_runtime as r
    rows=[dict(episode_id='episode-'+str(i),reset_id='reset-'+str(i),run_id='run',job_id=str(i),
        split='train',subject_id='p000439',capture_root=str(tmp_path/str(i)/'compact_multimodal'),
        run_result=dict(path=str(tmp_path/str(i)/'COMPLETE.json'),sha256=str(i)*64)) for i in (1,2)]
    manifest=r.write_new(tmp_path/'manifest.json',dict(schema=m.MANIFEST_SCHEMA,episodes=rows))
    def decode(row,base):
        sample=dict(states=np.zeros((1,17)),histories=np.zeros((1,2,17)),actions=np.zeros((1,20,7)),
            origins=np.zeros((1,7)),velocities=np.zeros((1,7)),observation_times=np.array([1.]),
            derived_index=dict(windows=[{}]))
        receipt=dict(native_reset_identity=dict(pid=41,sid=40,start_ticks='100'))
        return sample,receipt,{}
    monkeypatch.setattr(m,'read_episode',decode)
    result=r.read_bound(m.export_dataset(manifest,tmp_path/'out'))
    assert [o['positive_data'] for o in result['outcomes']]==[True,False]
    assert 'reset' in result['outcomes'][1]['error'] and result['windows']==1
