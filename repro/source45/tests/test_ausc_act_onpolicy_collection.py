"""Software lifecycle contracts only, never measured physical qualification."""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest
from test_ausc_given_contact_capture import new_inputs
from test_ausc_given_contact_handoff import FixtureTrajectory


def module():
    name='sim.ausc_policy.m2_act_onpolicy_collection'
    assert importlib.util.find_spec(name) is not None, 'Typed live collection API missing'
    return importlib.import_module(name)


def test_fixed_slots_and_foreign_prepared_reject_before_payload(monkeypatch,tmp_path):
    m=module();assert [r['slot_id'] for r in m.planned_requests()]==['dart-'+str(i).zfill(2) for i in (2,3,8,9,14,15,16,17)]
    assert all(r['split']=='train' for r in m.planned_requests())
    seen=[];monkeypatch.setattr(m.rt,'read_bound',lambda ref:seen.append(ref))
    with pytest.raises(ValueError):m.prepare(dict(m.PREPARED,bytes=1),{},tmp_path/'out')
    assert not seen


def query_binding(tmp_path):
    m=module();ref,_,_,meta=new_inputs(tmp_path)
    # Synthetic route binding is authenticated by the same production verifier;
    # only provenance filesystem fixtures are injected by individual tests.
    q=np.zeros(7);state=m.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    query=dict(measured_q=q.tolist(),measured_dq=q.tolist(),history=[state.tolist()]*2,history_times=[.95,1.],
        time_s=1.,physics_row=240,context=[meta['episode_id'],meta['target_id'],0,'FREE'],metadata=meta,
        reference_sha256=ref.sha256,hall_N=0.,physical_contact_N=0.,compression_m=0.)
    binding=dict(schema=m.BINDING_SCHEMA,metadata=meta,reference=ref,bootstrap=dict(schema=m.warm.COLLECTION_SCHEMA,
        collection_only=True,benchmark_model=False,optimizer_created=False),deployment={},configuration={},config_sha256=m.rt.digest({}))
    binding['live_witness']=dict(schema=m.LIVE_SCHEMA,query=deepcopy(query))
    return m,ref,query,binding


@pytest.mark.parametrize('phase',['FREE','APPROACH'])
@pytest.mark.parametrize('fault',['none','velocity','hall','contact','compression','clock','history','phase','split','binding'])
def test_live_validation_preserves_original_numeric_gates(tmp_path,monkeypatch,phase,fault):
    m,ref,q,b=query_binding(tmp_path);q['context'][-1]=phase;b['live_witness']['query']=deepcopy(q)
    monkeypatch.setattr(m,'validate_binding',lambda binding:binding)
    if fault=='velocity':q['measured_dq'][0]=.17500001
    elif fault=='hall':q['hall_N']=.2
    elif fault=='contact':q['physical_contact_N']=.2
    elif fault=='compression':q['compression_m']=.00015001
    elif fault=='clock':q['time_s']+=.001
    elif fault=='history':q['history'][0][0]+=.001
    elif fault=='phase':q['context'][-1]='ALIGN'
    elif fault=='split':q['metadata']['split']='validation'
    elif fault=='binding':b['metadata']=dict(b['metadata'],episode_id='other')
    b['live_witness']['query']=deepcopy(q)
    if fault=='none':np.testing.assert_array_equal(m.validate_live_collection_query(q,ref,b),np.zeros(7))
    else:
        with pytest.raises(ValueError):m.validate_live_collection_query(q,ref,b)


def test_final_destination_only_and_no_legacy_free_validator(tmp_path,monkeypatch):
    m,ref,q,b=query_binding(tmp_path);monkeypatch.setattr(m,'validate_binding',lambda binding:binding)
    monkeypatch.setattr(m.recovery,'propose_recovery',lambda *a,**k:pytest.fail('legacy FREE route'))
    canonical=np.asarray([[0.]*7,[.001]*7,[.002]*7])
    monkeypatch.setattr(m.rt.pose,'bound_json',lambda ref:dict(transfer_q=canonical.tolist()))
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    t=JointTrajectory(np.zeros(7),dt=1/240,**m.GIVEN_TRAJECTORY_LIMITS);before=m.handoff.input_snapshot(t)
    b['live_witness'].update(input=m.handoff.input_snapshot(t),output=m.handoff.output_snapshot(t),command_history=np.zeros((3,7)).tolist(),
        object_ids={k:id(v) for k,v in dict(trajectory=t,input=t.input,output=t.output,generator=t.generator).items()})
    planner=lambda start,goal,deadline:(np.vstack([start,goal]),dict(clear=True))
    wrong=m.propose_live_collection_recovery(q,t,np.zeros((3,7)),canonical[1],reference=ref,
        collection_binding=b,planner=planner,verify_command=lambda q:None)
    assert wrong['kind']=='abstain' and 'FINAL' in wrong['reason']
    result=m.propose_live_collection_recovery(q,t,np.zeros((3,7)),canonical[-1],reference=ref,
        collection_binding=b,planner=planner,verify_command=lambda q:None)
    assert result['kind']=='recovery_proposal'
    assert result['destination_q']==canonical[-1].tolist() and m.handoff.input_snapshot(t)==before


def test_all_eight_required_and_collection_never_benchmark_eligible():
    m=module();rows=[dict(request=r,status='qualified',windows=1,infrastructure_contamination=False,
        native_reset_identity=dict(pid=100+i,pgid=100+i,sid=100+i,start_ticks=str(i))) for i,r in enumerate(m.planned_requests())]
    gate=m.admission(rows);assert gate['data_optimization_eligible'] and not gate['benchmark_act_eligible']
    rows[-1]['status']='physical_negative'
    assert not m.admission(rows)['data_optimization_eligible']
    assert not m.admission(rows[:-1])['data_optimization_eligible']


def constructed(tmp_path,monkeypatch,klass=None,*,lifecycle=False,native_witness=False):
    """Real base construction, with a started synthetic compact capture."""
    from types import SimpleNamespace
    from sim.ausc_policy import expert_capture
    from test_ausc_policy_compact_expert_capture import compact_binding
    from test_ausc_policy_expert_capture import GEOMETRY,Env,Backend
    from sim.curobo_expert_v1.static_world import contract_sha
    from sim.ausc_policy.m2_given_contact_pose import GivenPoseProvider
    m,ref,query,binding=query_binding(tmp_path)
    monkeypatch.setattr(m,'validate_binding',lambda b:b)
    monkeypatch.setattr(m,'validate_bootstrap',lambda p,b:p)
    meta=binding['metadata']
    if native_witness:
        from sim.ausc_policy import m2_stop_residual_trajectory as residual
        meta.update({key:m.planned_requests()[0][key] for key in ('subject_id','target_id')})
        binding['configuration']={residual.CONFIG_KEY:deepcopy(residual.CONFIG)}
        binding['config_sha256']=m.rt.digest(binding['configuration'])
    if lifecycle:
        from test_ausc_given_contact_pose import bound
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        from sim.ausc_policy import m2_given_contact_pose as pose
        kin_actual=OpenArmKinematics(contact_frame=m.rt.geometry.contact_frame);goal=np.array([.5,0.,0.,0.,0.,0.,0.])
        full=kin_actual.tcp(goal);normal=-full[:3,2];point=full[:3,3]-normal*.03
        plane=dict(contact_plane_normal_world=normal.tolist(),contact_plane_radius_m=.023,plane_tilt_limit_deg=3.,tilt_relaxed=False)
        target=dict(name=meta['target_id'],position_world_m=point.tolist(),normal_world=normal.tolist(),**plane)
        patient=dict(identity=meta['subject_id'],split='train',asset_sha256='a'*64,non_patient_scene_unchanged=True,targets=[target])
        patient_ref=bound(tmp_path,'life-patient.json',patient)
        plan=dict(meta,success=True,name=meta['target_id'],position_world=point.tolist(),normal_world=normal.tolist(),
            command_normal_world=normal.tolist(),command_rotation_world=full[:3,:3].tolist(),world_from_left_base=np.eye(4).tolist(),
            patient_provenance=dict(identity=meta['subject_id'],asset_sha256='a'*64,binding=patient_ref),
            transfer_q=[np.zeros(7).tolist(),(goal/2).tolist(),goal.tolist()],lower_staging_q=(goal/2).tolist(),
            free_approach_start_gap_m=.03,contact_plane_normal_world=normal.tolist(),contact_plane_radius_m=.023,
            plane_tilt_limit_deg=3.,tilt_relaxed=False,selected_fit=dict(first_contact=plane))
        plan_ref=bound(tmp_path,'life-plan.json',plan)
        ref=pose.build_given_reference(dict(target,manifest_ref=patient_ref),plan_ref,patient_ref,np.eye(4),0.)
        meta=pose.route_metadata(meta,ref);meta['handoff_policy_id']=m.handoff.POLICY_ID
        meta.update(patient_asset_sha256=patient['asset_sha256'],installed_patient_receipt_sha256=patient_ref['sha256'])
        binding.update(metadata=meta,reference=ref)
    env=Env();env.base=np.eye(4);env.bodies=[];env.out=tmp_path/'fresh_expert'
    env.value.update(time_s=0.,compression_m=0.,hall_N=0.,rotation_world=np.eye(3),tcp_base=np.zeros(3))
    raw=compact_binding();raw['episode_id']=meta['episode_id'];raw['compact'].update(metadata=meta,
        estimated_target_base=ref.pose_base.tolist(),target_confidence=1.,target_estimate_source_id='given-contact-'+ref.sha256,
        max_frames=2000 if native_witness else 1000)
    raw['compact']['code_bindings'].update({name:r['sha256'] for name,r in m._capture_source_bindings().items()})
    geometry=deepcopy(GEOMETRY)
    if native_witness:
        from test_ausc_policy_retainer_witness import contract_fixture
        witness_dir=tmp_path/'cpu-witness';witness_dir.mkdir()
        meta['native_retainer_audit']=contract_fixture(witness_dir)[0]
        geometry['native_retainer_audit']=meta['native_retainer_audit']
    capture=expert_capture.create_expert_capture(tmp_path/'capture',binding=raw,geometry_metadata=geometry,
        source_paths=[r['path'] for r in m._capture_source_bindings().values()])
    env.raw_capture=capture;capture.start(env,backend_factory=Backend)
    contract_value=dict(body_reference_poses={},finger_specs={},translation_tolerance_m=.001,rotation_tolerance_rad=.001,
        finger_joint_numerical_tolerance_m=.001)
    bundle=dict(contract=contract_value,contract_sha256=contract_sha(contract_value),geometry={})
    checker=SimpleNamespace(static_bundle=bundle,static_installation=dict(contract_sha256=bundle['contract_sha256'],geometry_sha256={}),
        _static_containment_installed=True)
    provider=GivenPoseProvider(ref,meta['episode_id'],meta)
    kin=kin_actual if lifecycle else SimpleNamespace(lower_limits=np.full(7,-3.),upper_limits=np.full(7,3.))
    class Predictor:
        context=None
        def __call__(self,q,p,c,*,metadata,history,history_times):
            return m.contract.JointActionBlock('block_relative_joint',np.zeros((20,7)),q,metadata)
        def close(self):return dict(schema=m.CPU_EXIT_SCHEMA,normal_exit=True,reaped=True,exit_code=0,signals=[])
    kwargs=dict(metadata=meta,episode_id=meta['episode_id'],estimate_provider=provider,static_world_bundle=bundle)
    if klass is None:
        binding['deployment']['source_inventory']=m.sources()
        deployment_ref=m.rt.write_new(tmp_path/'pilot_cpu_deployment.json',binding['deployment'])
        binding['bootstrap_ref']=m.rt.write_new(tmp_path/m.BOOTSTRAP_FILENAME,dict(schema=m.BOOTSTRAP_SIDECAR_SCHEMA,
            deployment=deployment_ref,source_inventory=binding['deployment']['source_inventory'],bootstrap=binding['bootstrap']))
        controller=m.OnPolicyTeacherCollection(env,kin,checker,predictor=Predictor(),bootstrap_receipt=binding['bootstrap'],
            bootstrap_ref=binding['bootstrap_ref'],collection_binding=binding,**kwargs)
    else:
        with pytest.raises(ValueError,match='Expert capture cannot label'):
            klass(env,kin,checker,predictor=Predictor(),policy_receipt={},**kwargs)
        controller=None
    return m,controller,env,capture


def test_real_constructor_keeps_started_capture_attached_and_ordinary_act_rejects(tmp_path,monkeypatch):
    m,c,env,capture=constructed(tmp_path,monkeypatch)
    assert env.raw_capture is capture is c._collection_capture and capture.writer is not None and capture.camera is not None
    assert c.joint_speed_limit==.175 and c.estimate_provider.bound['physics_row']==0
    capture.close()
    from sim.ausc_policy.m2_given_contact_controller import GivenContactACT
    from sim.ausc_policy.m2_act_controller import M2ACTController
    for i,klass in enumerate((GivenContactACT,M2ACTController)):
        d=tmp_path/str(i);d.mkdir()
        # The direct old controller also retains its original given-task guard.
        if klass is M2ACTController:
            class OldGiven(M2ACTController):task_id=m.rt.pose.TASK_ID
            klass=OldGiven
        _,_,env,capture=constructed(d,monkeypatch,klass=klass)
        assert env.raw_capture is capture;capture.close()


def test_factory_closes_cpu_child_on_real_constructor_failure(monkeypatch,tmp_path):
    m=module();closed=[];binding=dict(deployment={},metadata={},reference=None)
    monkeypatch.setattr(m,'preflight',lambda *a,**k:binding)
    monkeypatch.setattr(m,'validate_bootstrap',lambda p,b:p)
    class Bridge:
        bootstrap={};bootstrap_ref={}
        def __init__(self,*a,**k):pass
        def close(self):closed.append(True)
    monkeypatch.setattr(m,'CPUBridge',Bridge)
    with pytest.raises((KeyError,ValueError,AttributeError,TypeError)):m.make_controller(None,config={},output=tmp_path)
    assert closed==[True]


def test_first_grid_trigger_and_cumulative_deadline_do_not_reset(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m,c,env,capture=constructed(tmp_path,monkeypatch)
    c.manager=SimpleNamespace(config=SimpleNamespace(precontact_gap_m=.04));c._l2_started_at=1.
    c._history=m.stage.CausalHistory(1.);state=np.zeros(17);state[9]=-.1
    c._history.frames=[(40.95,state.copy()),(41.,state.copy())]
    env.rows=[{}]*9840;env.value['time_s']=41.
    assert c._eligible({})
    env.rows=env.rows[:-1];assert not c._eligible({})
    env.rows=[{}]*9840;env.value['time_s']=40.999
    assert not c._eligible({})
    env.value['time_s']=121.
    with pytest.raises(ValueError,match='120 second'):c._deadline()
    assert c._l2_started_at==1.;capture.close()


def test_capture_boundary_api_exists_and_refuses_unexecuted_preview():
    m=module();assert callable(m.verify_captured_boundary)
    with pytest.raises((ValueError,KeyError)):m.verify_captured_boundary({},[],[],{}, {})


def produced_collection(tmp_path,monkeypatch,*,stop_residual=False,full_cycle=False,native_like=False):
    """Physics leaf and contact cycle are invented; production ownership is real."""
    from types import SimpleNamespace
    from dataclasses import replace
    from sim.ausc_policy import fresh_expert, m2_given_contact_controller as boundary
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    m,c,env,capture=constructed(tmp_path,monkeypatch,lifecycle=True,native_witness=full_cycle)
    from sim.ausc_policy import m2_stop_residual_trajectory as residual
    if stop_residual:c.trajectory_factory=residual.StopResidualTrajectory
    ref=c.reference;plan=m.rt.pose.bound_json(ref.snapshot()['sources']['fresh_plan']);seen=[]
    env.last_contacts={k:0. for k in m.ALL_FORCE_GROUPS}
    c.target=SimpleNamespace(name=c.metadata['target_id'],position_base=ref.pose_base[:3,3],position_world=ref.pose_base[:3,3])
    c.normal_base=c.normal=ref.normal_base;c.rotation_base=ref.pose_base[:3,:3];c.sequence_index=0
    c.manager=SimpleNamespace(config=replace(c.config,precontact_gap_m=.03),state=SimpleNamespace(phase=Phase.APPROACH,failure_reason=None))
    c.posture_limits=SimpleNamespace(evaluate=lambda *a,**k:dict(clear=True))
    c.checker.check=lambda *a,**k:dict(clear=True,minimum_distance_m=.1)
    c._plan_inter_point_bridge=lambda start,goal,deadline:(np.vstack([start,goal]),dict(clear=True))
    def leaf(self,command,subphase,active=False,recovery=False):
        assert self.env.raw_capture is capture and capture.writer is not None
        n=len(env.rows);prior=env.value['q'].copy();old=env.value.get('tcp_base',np.zeros(3)).copy();p=self.kin.tcp(command)
        measured=command.copy()
        if native_like:
            from test_ausc_act_native_pose_consistency import native_like_pose
            measured=command.astype(np.float32).astype(float)
            env.value['compression_m']=.002 if subphase=='hold' else float(np.float32(-1.297662244e-8))
            p=native_like_pose(self.kin,measured,env.base,env.value['compression_m'])
        env.value.update(q=measured,dq=(np.array([.1,0.,0.,0.,0.,0.,0.]) if n+1==9840 else (measured-prior)*240.),time_s=(n+1)/240.,tcp_base=p[:3,3].copy(),
            tcp_world=p[:3,3].copy(),rotation_world=p[:3,:3].copy(),axis_world=p[:3,2].copy())
        env.rows.append(dict({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in env.state().items()},q_command=command.tolist(),subphase=subphase,phase=self.manager.state.phase.value,
            target=c.metadata['target_id'],physical_violations=[],**{k+'_N':0. for k in m.ALL_FORCE_GROUPS},tcp_speed_m_s=float(np.linalg.norm(p[:3,3]-old)*240.),**self.metrics(env.state())))
        if full_cycle:
            hold=subphase=='hold';env.value.update(hall_N=4. if hold else 0.)
            if not native_like:env.value['compression_m']=.002 if hold else 0.
            env.rows[-1].update(hall_N=env.value['hall_N'],compression_m=env.value['compression_m'],diaphragm_N=4. if hold else 0.,
                native_retainer_audit=c.metadata['native_retainer_audit'],retainer_front_audit=dict(schema='retainer_front_audit_v3',
                    native_retainer_audit=c.metadata['native_retainer_audit']))
            if self.control_owner=='ANALYTIC_L3_ALIGN':self._alignment_audit.observe(p,'observed')
        self.q=command.copy();capture.after_physics(env,n)
        if self.control_owner in ('ACT_L2','TEACHER_L2'):seen.append((self.control_owner,self.estimate_provider.stage.phase,n+1))
        return env.state()
    monkeypatch.setattr(fresh_expert.SkillExpert,'_tick',leaf)
    pose0=c.kin.tcp(c.q)
    if native_like:
        from test_ausc_act_native_pose_consistency import native_like_pose
        env.value['compression_m']=float(np.float32(-1.297662244e-8))
        pose0=native_like_pose(c.kin,c.q,env.base,env.value['compression_m'])
    env.value.update(tcp_base=pose0[:3,3].copy(),tcp_world=pose0[:3,3].copy(),rotation_world=pose0[:3,:3].copy(),axis_world=pose0[:3,2].copy())
    c._move_to_prepose(plan)
    intervention=c._intervention
    assert intervention['query']['context'][-1]=='FREE' and intervention['time_s']==41.
    assert intervention['physics_row']==9840 and intervention['live_before']==intervention['live_after']
    assert intervention['live_before']['history_id']==id(c._history) or c._history is None  # parking may end L2 history
    assert c._intervention_attempts==1 and c._teacher_completed and c.control_owner=='L3_ENTRY'
    assert any(owner=='TEACHER_L2' and phase=='APPROACH' for owner,phase,_ in seen)
    assert c.l2_handoffs[-1]['l2_coarse_reached'] and c.l2_handoffs[-1]['l2_handoff_success']
    assert max(r['physics_row_start'] for r in c.l2_actions)<9840
    assert c._l2_started_at==1. and env.state()['time_s']<121.
    from test_ausc_given_contact_controller import gates
    def alignment_gates():
        g=gates(ref,c.kin.tcp(c.q));g['plane']=dict(contact_plane_normal_world=ref.normal_base.tolist(),
            contact_plane_radius_m=.023,plane_tilt_limit_deg=3.,tilt_relaxed=False)
        return g
    c._alignment_gates=alignment_gates
    def align(*args):
        u=np.arange(1,481)/480;blend=10*u**3-15*u**4+6*u**5;goal=np.asarray(plan['transfer_q'])[-1]
        commands=c.q+(goal-c.q)*blend[:,None];commands=np.vstack([commands,np.tile(goal,(3,1))])
        report=dict(schema='m2_given_contact_l3_align_v1',bounds=boundary.BOUNDS,reference_sha256=ref.sha256,
            reference_revision=0,all_240hz_commands_verified=True,final_stationary_stop_included=True,
            duration_s=len(commands)/240.,planning_cpu_s=0.)
        return boundary.AlignmentProposal(commands,report)
    monkeypatch.setattr(boundary,'propose_l3_alignment',align)
    analytic=[]
    def contact(self):
        analytic.append(self.control_owner)
        if full_cycle:
            for phase,subphase,ticks in ((Phase.HOLD,'hold',720),(Phase.RETRACT,'normal_release_confirm',48),
                                        (Phase.DONE,'home_settle',36)):
                self.manager.state.phase=phase
                for _ in range(ticks):self._tick(self.q.copy() if subphase!='home_settle' else np.zeros(7),subphase)
    monkeypatch.setattr(fresh_expert.FreshExpert,'_contact_cycle',contact)
    c._contact_cycle()
    assert analytic==['ANALYTIC_L3'] and c.l2_handoffs[-1]['l3_completed']
    assert env.raw_capture is capture;capture.close()
    import gzip,json
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:audit=[json.loads(line) for line in stream]
    if stop_residual:
        receipt=c.l2_handoffs[-1]
        residual.verify_native_parking(dict(handoff_policy_id=m.handoff.POLICY_ID,
            runtime_sources_sha256=receipt['parking_preview']['source_sha256']),receipt,audit,c.metadata,ref,
            c.manager.config,config={residual.CONFIG_KEY:deepcopy(residual.CONFIG)},kin=c.kin)
    assert {r['actual_control_owner'] for r in audit[240:9840]}=={'ACT_L2'}
    assert audit[9840]['actual_control_owner']=='TEACHER_L2'
    assert all(e['kind']!='recovery_teacher' for e in c._collection_events if e['physics_row_start']<9840)
    sidecar=dict(schema=m.SIDECAR_SCHEMA,completed=True,interventions=1,native_steps=len(audit),episode_id=c.episode_id,
        reference_sha256=ref.sha256,config_sha256=c.collection_binding['config_sha256'],deployment=c.collection_binding['deployment'],
        learner_prefix_executed=True,benchmark_act_eligible=False,bootstrap=c.bootstrap_ref,cpu_child_exit=c.predictor.close(),
        intervention=c._intervention,events=c._collection_events)
    records=list(m.data.iter_training_records(capture.output/'compact'))
    checked=m.verify_captured_boundary(sidecar,audit,records,c.metadata,c.collection_binding)
    assert checked['first_eligible_boundary'] and checked['retained_qva_replayed']
    assert sidecar['intervention']['query']['measured_dq']==[.1,0.,0.,0.,0.,0.,0.]
    accepted=[]
    for fault in ('same_max_dq_swap','output_position','output_velocity','output_acceleration',
                  'missing_capture_record','missing_capture_output','stale_capture_row','stale_capture_time'):
        forged=deepcopy(sidecar);raw=deepcopy(audit);intervention=forged['intervention'];cut=intervention['physics_row']
        if fault=='same_max_dq_swap':
            for query in (intervention['query'],intervention['live_before']['query'],intervention['live_after']['query'],intervention['proposal']['query']):
                query['measured_dq']=[0.,-.1,0.,0.,0.,0.,0.]
            intervention['proposal']['query_sha256']=m.rt.digest(intervention['query'])
        elif fault.startswith('output_'):
            key={'output_position':'new_position','output_velocity':'new_velocity','output_acceleration':'new_acceleration'}[fault]
            for snapshot in (intervention['live_before'],intervention['live_after']):snapshot['output'][key][0]=.123
        else:
            if fault=='missing_capture_record':raw[cut-1].pop('onpolicy_collection_boundary',None)
            else:
                witness=raw[cut-1].setdefault('onpolicy_collection_boundary',dict(physics_row=cut,time_s=cut/240.))
                if fault=='missing_capture_output':witness.pop('output',None)
                elif fault=='stale_capture_row':witness['physics_row']-=12
                elif fault=='stale_capture_time':witness['time_s']-=.05
        try:m.verify_captured_boundary(forged,raw,records,c.metadata,c.collection_binding)
        except (ValueError,KeyError):pass
        else:accepted.append(fault)
    assert not accepted, 'Collection verifier accepted tampering: '+str(accepted)
    return m,c,env,capture,audit,sidecar,records


@pytest.mark.parametrize('stop_residual',[False,True])
def test_genuine_free_40s_actual_oracle_teacher_phase_parking_and_l3(tmp_path,monkeypatch,stop_residual):
    produced_collection(tmp_path,monkeypatch,stop_residual=stop_residual)


def test_cumulative_collection_clock_includes_prefix_teacher_and_parking():
    m=module();assert callable(m.cumulative_l2)
    audit=[dict(actual_control_owner='PREPARATION')]*240+[dict(actual_control_owner='ACT_L2')]*9600
    audit += [dict(actual_control_owner='TEACHER_L2')]*120+[dict(actual_control_owner='L2_STOP')]*120
    run=dict(l2_handoffs=[dict(start_physics_row=240,start_time_s=1.,parking_end_physics_row=len(audit))])
    result=m.cumulative_l2(run,audit)
    assert result['duration_s']==41. and result['prefix_steps']==9600
    long=audit[:9840]+[dict(actual_control_owner='TEACHER_L2')]*20000
    run['l2_handoffs'][0]['parking_end_physics_row']=len(long)
    with pytest.raises(ValueError,match='120'):m.cumulative_l2(run,long)


def test_negative_collection_reader_is_explicit_and_never_a_positive_export():
    m=module();assert callable(m.read_negative)
    with pytest.raises((ValueError,KeyError)):m.read_negative({},None,{})


def test_physically_consistent_but_fabricated_h2_is_not_the_certified_live_query(tmp_path,monkeypatch):
    m,ref,q,b=query_binding(tmp_path);monkeypatch.setattr(m,'validate_binding',lambda b:b)
    b['live_witness']=dict(schema='m2_act_onpolicy_live_state_certificate_v1',query=deepcopy(q))
    previous=np.asarray(q['measured_q'])+.001
    q['history'][0]=m.contract.encode_state(previous,ref.pose_base,1.,metadata=q['metadata']).tolist()
    with pytest.raises(ValueError,match='live'):m.validate_live_collection_query(q,ref,b)


@pytest.mark.parametrize('fault',['none','short_hold','hall','compression','lip','contact_rim','carriage_other','frame','forbidden','release','home'])
def test_collection_physical_kernel_keeps_all_force_hold_release_home_gates(tmp_path,monkeypatch,fault):
    from test_ausc_given_contact_dataset import physical_fixture
    from sim.ausc_policy import m2_stop_residual_trajectory as residual
    m=module();run,audit,meta=physical_fixture(tmp_path,monkeypatch)
    meta['handoff_policy_id']=m.handoff.POLICY_ID
    config={residual.CONFIG_KEY:deepcopy(residual.CONFIG)};run['runtime_sources_sha256']=residual._source_hashes()
    # Only numerical parking is isolated here; the real owned lifecycle above
    # exercises it. This is the exact complete physical kernel used by decoder.
    monkeypatch.setattr(residual,'verify_native_parking',lambda *a,**k:None)
    if fault=='short_hold':audit[500]['measurements']['diaphragm_N']=0.
    elif fault=='hall':audit[500]['measurements']['hall_N']=6.6
    elif fault=='compression':audit[500]['measurements']['compression_m']=.0061
    elif fault in ('lip','contact_rim','carriage_other','frame','forbidden'):audit[500]['measurements'][fault+'_N']=.001
    elif fault=='release':audit[-10]['measurements']['diaphragm_N']=.3
    elif fault=='home':audit[-1]['controller_after']['subphase']='retract'
    if fault=='none':residual.verify_native_physical(run,audit,meta,config=config)
    else:
        with pytest.raises(ValueError):residual.verify_native_physical(run,audit,meta,config=config)


def test_boundary_capture_uses_post_step_latch_without_observing_or_changing_history(tmp_path,monkeypatch):
    """A synthetic measured edge checks capture timing, never physics success."""
    from types import SimpleNamespace
    from sim.ausc_policy import expert_capture
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    m,c,env,capture=constructed(tmp_path,monkeypatch,lifecycle=True)
    c.target=SimpleNamespace(name='L1');c.manager=SimpleNamespace(state=SimpleNamespace(phase=Phase.APPROACH))
    c.control_owner='ACT_L2';c.actor_phase='FREE';c._history=m.stage.CausalHistory(0.)
    c.trajectory=JointTrajectory(np.zeros(7),dt=1/240,**m.GIVEN_TRAJECTORY_LIMITS)
    context=(c.episode_id,'L1',0,'FREE');c._issue_intent(np.zeros(7),context,'given_l2_history_wait')
    history_before=deepcopy(c._history.frames);provider_before=deepcopy(vars(c.estimate_provider.stage))
    goal=np.asarray(m.rt.pose.bound_json(c.reference.snapshot()['sources']['fresh_plan'])['transfer_q'])[-1]
    for i in range(12):
        command=c.trajectory.step();evidence=c._dispatcher().snapshot(time_s=i/240.,physics_row=i)
        evidence['live_target_after_step_q']=evidence['live_target_q'].copy();c._current_intent_audit=evidence
        before=expert_capture.controller_labels(c,'given_l2_history_wait',False,False)
        env.advance();env.value.update(q=goal.copy() if i==11 else np.zeros(7),dq=np.array([.1,0.,0.,0.,0.,0.,0.]))
        pose=c.kin.tcp(env.value['q']);env.value.update(tcp_base=pose[:3,3],rotation_world=pose[:3,:3])
        env.rows[-1].update(q=env.value['q'].tolist(),dq=env.value['dq'].tolist(),target='L1',q_command=command.tolist(),
            subphase='given_l2_history_wait',physical_violations=[])
        capture.after_physics(env,i)
        after=expert_capture.controller_labels(c,'given_l2_history_wait',False,False)
        capture.finish_tick(env,i,before,after,clean_command=dict(q_command=command.tolist(),issued_at_s=i/240.,
            endpoint_at_s=(i+1)/240.,readback_at_s=(i+1)/240.,source_id='onpolicy_collection_act_native_command_v1'),
            control_owner='L2',collection_controller=c)
    assert before['actor']['phase'] is None and after['actor']['phase'] is None  # WAIT labels stay excluded
    assert capture._actor_stage.phase=='APPROACH'
    assert c.actor_phase=='FREE' and c._history.frames==history_before and vars(c.estimate_provider.stage)==provider_before
    capture.close()
    import gzip,json
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:audit=[json.loads(line) for line in stream]
    assert all('onpolicy_collection_boundary' not in e for e in audit[:-1])
    witness=audit[-1]['onpolicy_collection_boundary']
    assert witness['context'][-1]=='APPROACH' and witness['physics_row']==12 and witness['time_s']==.05
    assert witness['measured_dq']==[.1,0.,0.,0.,0.,0.,0.]
    assert witness['input']==m.handoff.input_snapshot(c.trajectory) and witness['output']==m.handoff.output_snapshot(c.trajectory)
    assert witness['object_ids']=={k:id(v) for k,v in dict(trajectory=c.trajectory,input=c.trajectory.input,
        output=c.trajectory.output,generator=c.trajectory.generator).items()}


def bootstrap_handshake(tmp_path,monkeypatch,*,fault='none',collection=None):
    """Real subprocess/serve/framing; only model and Linux/provenance leaves are synthetic."""
    import json,os,sys
    from pathlib import Path
    from sim.ausc_policy import process_guard
    if collection is None:m,ref,query,binding=query_binding(tmp_path)
    else:
        m=module();binding=collection.collection_binding;ref=binding['reference']
        state=m.contract.encode_state(np.zeros(7),ref.pose_base,1.,metadata=binding['metadata']).tolist()
        query=dict(history=[state,state],history_times=[.95,1.],context=[binding['metadata']['episode_id'],binding['metadata']['target_id'],0,'FREE'])
    initialization=dict(normalization={'synthetic':'unchanged'},posture_head={'synthetic':'head'},
        tensor_proofs={f'tensor.{i}':dict(shape=[1],dtype='float32',bytes=4,sha256=f'{i:064x}') for i in range(54)},
        parent_model=deepcopy(m.warm.PARENT_MODEL),historical_optimization={'synthetic':'completed20000-selected17000'})
    init_ref=m.rt.write_new(tmp_path/'initialization.json',initialization)
    inventory=m.sources()
    dep=dict(initialization=init_ref,model=dict(checkpoint=dict(path='/synthetic/tensors.pt',bytes=4,sha256='a'*64)),
        metadata=binding['metadata'],source_inventory=inventory,episode_id=binding['metadata']['episode_id'],
        source_episode_id='synthetic-original',source_config=dict(sha256='b'*64),request=m.planned_requests()[0])
    binding.update(deployment=dep,initialization=initialization)
    software=dict(schema=m.rt.RUNTIME_SCHEMA,prefix=m.rt.VENDOR_ROOT+'/.venv',python_version='synthetic CPU',
        python=dict(path='/synthetic/python',bytes=1,sha256='c'*64),
        vendor=dict(root=m.rt.VENDOR_ROOT,upstream_commit=m.rt.UPSTREAM_COMMIT,
            files={name:dict(path=m.rt.VENDOR_ROOT+'/'+name,bytes=1,sha256='d'*64)
                   for name in [*(f'src/lerobot/synthetic_{i:04}.py' for i in range(880)),'pyproject.toml','uv.lock']}),
        dependencies=[dict(name='synthetic',version='1',metadata_location=m.rt.VENDOR_ROOT+'/.venv/lib/synthetic.dist-info',
            files={},payload_source={'kind':'synthetic'},payloads={f'package/synthetic_{i:05}.py':dict(reference=dict(
                path=m.rt.VENDOR_ROOT+f'/.venv/lib/package/synthetic_{i:05}.py',bytes=1,sha256='e'*64)) for i in range(12000)})],
        source_bindings={str(Path(mod.__file__).resolve()):m.rt.shared.sha256(mod.__file__)
            for mod in (m.rt.shared,m.rt.given,m.rt.pose,m.rt.handoff,m.rt)})
    packet=dict(schema=m.warm.COLLECTION_SCHEMA,collection_only=True,benchmark_model=False,optimizer_created=False,
        initialization=init_ref,**initialization,source_evidence=dict(parent=deepcopy(m.warm.SOURCE41),
            current=dict(software=software,helper=m.rt.reference(m.warm.__file__))),
        equivalence=dict(schema=m.warm.EQUIVALENCE_SCHEMA,passed=True,collection_only=True,optimizer_created=False,
            initialization=init_ref,initial_tensor_proofs=deepcopy(initialization['tensor_proofs'])))
    assert len(json.dumps(packet).encode())>1_000_000 and len(software['vendor']['files'])==882
    fixture=m.rt.write_new(tmp_path/'fixture.json',dict(packet=packet,initialization=initialization,query=query,inventory=inventory))
    execution=dict(device='cpu',parameter_devices=['cpu'],parameter_dtypes=['torch.float32'],default_dtype='torch.float32',
        intraop_threads=2,interop_threads=1,deterministic_algorithms=True,matmul_precision='highest',matmul_tf32=False,
        cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8',
        torch_version='synthetic CPU',cuda_version=None,device_name='cpu')
    script=tmp_path/'child.py'
    script.write_text('''import json,sys,os
from pathlib import Path
from sim.ausc_policy import m2_act_onpolicy_collection as m
sys.platform='linux'
fixture=json.loads(Path(sys.argv[1]).read_text());fault=sys.argv[2]
dep_ref=m.rt.reference(sys.argv[3]);dep=m.rt.read_bound(dep_ref)
ref=m.rt.pose.require_route(dep['metadata'])
binding=dict(schema=m.BINDING_SCHEMA,deployment=dep,initialization=fixture['initialization'],reference=ref)
m._validate_deployment=lambda d:({},fixture['initialization'],ref)
m.preflight=lambda *a,**k:binding
m.sources=lambda:fixture['inventory']
m.warm.make_collection_policy=lambda r:(None,None,fixture['packet'])
m.rt.infer=lambda *a,**k:m.np.full((1,20,7),.001)
original=m.rt.write_new
written={}
def write(path,value):
    result=original(path,value)
    if Path(path).name=='collection_cpu_bootstrap.json':
        written.update(path=path,reference=result)
    return result
def execution(*a,**k):
    # Mutate AFTER the child's own sidecar check; parent must reject READY.
    path=Path(written['path']);result=written['reference'];value=json.loads(path.read_text())
    if fault.startswith('bad_') or fault=='wrong_deployment':
        if fault=='wrong_deployment':value['deployment']['sha256']='0'*64
        if fault=='bad_schema':value['schema']='untyped'
        if fault=='bad_packet_schema':value['bootstrap']['schema']='untyped'
        if fault=='bad_inventory':value['source_inventory']={}
        if fault=='bad_initialization':value['bootstrap']['initialization']={}
        if fault=='bad_head':value['bootstrap']['posture_head']={}
        if fault=='bad_norm':value['bootstrap']['normalization']={}
        if fault=='bad_tensor':value['bootstrap']['tensor_proofs']['tensor.0']['sha256']='f'*64
        if fault=='bad_source':value['bootstrap']['source_evidence']['current']['helper']['sha256']='f'*64
        if fault=='bad_software_missing':value['bootstrap']['source_evidence']['current'].pop('software')
        if fault=='bad_software_sources':value['bootstrap']['source_evidence']['current']['software']['source_bindings']={}
        path.write_text(json.dumps(value));result.update(m.rt.reference(path))
    if fault=='corrupt':path.write_text('{}')
    if fault=='missing':path.unlink()
    if fault=='wrong_bytes':result['bytes']+=1
    if fault=='wrong_sha':result['sha256']='0'*64
    if fault=='wrong_path':
        other=path.with_name('other_bootstrap.json');path.rename(other);result.update(m.rt.reference(other))
    if fault=='alias':
        other=path.with_name('aliased_bootstrap.json');path.rename(other);path.symlink_to(other)
    if fault=='hardlink':os.link(path,path.with_name('hardlink.json'))
    if fault=='oversize_ref':result['padding']='x'*140000
    settings=json.loads(sys.argv[4])
    if fault=='bad_execution':settings['intraop_threads']=3
    return settings
m.rt.write_new=write
m.rt.execution_settings=execution
if fault=='bad_identity':m.os.getppid=lambda:0
if fault=='bad_ready_source':m.sources=lambda:{}
m.serve(dep_ref)
''')
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(m,'_validate_deployment',lambda d:({},initialization,ref))
    monkeypatch.setattr(m,'preflight',lambda *a,**k:binding)
    # /proc is the Linux-only leaf; transport and child identity values remain real.
    monkeypatch.setattr(process_guard,'process_identity',lambda pid:dict(pid=pid,pgid=os.getpgrp(),sid=os.getsid(0),start_ticks='fixture',state='R'))
    real_popen=m.pipe.subprocess.Popen;children=[]
    def launch(command,**kwargs):
        process=real_popen([sys.executable,'-u','-B',str(script),fixture['path'],fault,
            command[command.index('--deployment')+1],json.dumps(execution)],**kwargs)
        children.append(process);return process
    monkeypatch.setattr(m.pipe.subprocess,'Popen',launch)
    return m,dep,binding,packet,query,children


def test_production_serve_large_bootstrap_stays_off_wire_and_predicts_full_h20(tmp_path,monkeypatch):
    import json
    m,dep,binding,packet,query,children=bootstrap_handshake(tmp_path,monkeypatch)
    bridge=m.CPUBridge(dep,output=tmp_path/'case')
    try:
        assert bridge.bootstrap==packet and 'bootstrap' not in bridge.runtime
        assert bridge.runtime['bootstrap_ref']==bridge.bootstrap_ref
        assert m.rt.read_bound(bridge.bootstrap_ref)['bootstrap']==packet
        assert bridge.bootstrap_ref['path']==str(tmp_path/'case/collection_cpu_bootstrap.json')
        assert len(json.dumps(bridge.runtime).encode())<131072
        bridge.context=dict(observed_at_s=1.,physics_row=240,context=query['context'],source_id='given-contact-'+binding['reference'].sha256)
        block=bridge(np.zeros(7),binding['reference'].pose_base,1.,metadata=dep['metadata'],
            history=np.asarray(query['history']),history_times=np.asarray(query['history_times']))
        np.testing.assert_array_equal(block.values,np.full((20,7),.001))
    finally:exit=bridge.close()
    assert exit['normal_exit'] and exit['reaped'] and not exit['signals']
    trace=[json.loads(line) for line in (tmp_path/'case/pilot_cpu_ipc.jsonl').read_text().splitlines()]
    assert all(len(json.dumps(row).encode())<131072 for row in trace)
    assert 'dependencies' not in json.dumps(trace) and 'tensor_proofs' not in json.dumps(trace)
    assert len(children)==1 and children[0].poll()==0


@pytest.mark.parametrize('fault',['corrupt','missing','wrong_path','wrong_bytes','wrong_sha','alias','hardlink','wrong_deployment',
    'bad_schema','bad_packet_schema','bad_inventory','bad_initialization','bad_head','bad_norm',
    'bad_tensor','bad_source','bad_software_missing','bad_software_sources','oversize_ref','bad_identity','bad_ready_source','bad_execution'])
def test_production_bootstrap_handshake_rejects_and_reaps_child(tmp_path,monkeypatch,fault):
    import json
    m,dep,_,_,_,children=bootstrap_handshake(tmp_path,monkeypatch,fault=fault)
    bridge=None
    try:
        with pytest.raises((ValueError,FileNotFoundError)):bridge=m.CPUBridge(dep,output=tmp_path/'case')
    finally:
        if bridge is not None:bridge.close()
    assert len(children)==1 and children[0].poll() is not None
    assert m.rt.read_bound(m.rt.reference(tmp_path/'case/pilot_cpu_exit.json'))['reaped'] is True
    trace=(tmp_path/'case/pilot_cpu_ipc.jsonl').read_text()
    if fault not in ('oversize_ref','bad_identity','bad_ready_source','bad_execution'):
        assert json.loads(trace.splitlines()[0])['event']=='ready'  # Parent performed the rejection.
    if fault=='oversize_ref':assert 'Bounded collection CPU response required' in (tmp_path/'case/pilot_cpu_stderr.log').read_text()


def test_real_collection_factory_and_persisted_decoder_use_one_bootstrap_reference(tmp_path,monkeypatch):
    import json
    m=module();real_validator=m.validate_bootstrap
    case=tmp_path/'case';case.mkdir()
    m,original,env,capture=constructed(case,monkeypatch)
    # Remove only the small invented constructor fixtures; the production
    # bridge must create its own fresh deployment and bootstrap sidecar.
    (case/'pilot_cpu_deployment.json').unlink();(case/m.BOOTSTRAP_FILENAME).unlink()
    fixture=tmp_path/'fixture';fixture.mkdir()
    monkeypatch.setattr(m,'validate_bootstrap',real_validator)
    m,dep,binding,packet,_,children=bootstrap_handshake(fixture,monkeypatch,collection=original)
    controller=m.make_controller(env,original.kin,original.checker,config={},output=case,
        static_world_bundle=original.checker.static_bundle)
    assert controller.env.raw_capture is capture and controller.bootstrap_receipt==packet
    assert controller.bootstrap_ref==controller.predictor.bootstrap_ref
    # No physics is asserted: isolate the outer result/sidecar persistence seam.
    monkeypatch.setattr(m.GivenContactExpert,'run',lambda *a,**k:dict(success=True))
    controller._intervention_attempts=1;controller._teacher_completed=True;controller.l2_queries=[{'synthetic':True}]
    try:result=controller.run([])
    finally:
        controller.close_bridge();capture.close()
    m.tag_result(result,controller,{m.ROUTE_KEY:dep})
    sidecar=m.rt.read_bound(result['onpolicy_collection'])
    assert result['collection_bootstrap']==sidecar['bootstrap']==controller.bootstrap_ref
    assert len(json.dumps(result).encode())<131072 and len(json.dumps(sidecar).encode())<131072
    assert len(list(case.glob(m.BOOTSTRAP_FILENAME)))==1 and controller.bootstrap_ref['bytes']>1_000_000
    result.update(physical_validation=True,errors=[],controller_motion_complete=True,compact_capture_complete=True,
        runtime_sources_sha256={r['path']:r['sha256'] for r in dep['source_inventory']['files'].values()})
    freeze=dict(code_commit=dep['source_inventory']['code_commit'])
    m.validate_capture_context(result,{},freeze,binding)
    assert binding['bootstrap']==packet and binding['bootstrap_ref']==controller.bootstrap_ref
    with pytest.raises(ValueError,match='location'):
        m.load_bootstrap(controller.bootstrap_ref,binding,output=fixture)
    bad=deepcopy(result);bad['collection_bootstrap']['sha256']='0'*64
    with pytest.raises(ValueError,match='references differ'):m.validate_capture_context(bad,{},freeze,binding)
    assert children[0].poll()==0


def test_production_collection_bootstrap_requires_fresh_sidecar(tmp_path,monkeypatch):
    m,dep,_,_,_,children=bootstrap_handshake(tmp_path,monkeypatch)
    case=tmp_path/'case';case.mkdir();path=case/m.BOOTSTRAP_FILENAME;path.write_text('existing immutable fixture')
    with pytest.raises(ValueError):m.CPUBridge(dep,output=case)
    assert path.read_text()=='existing immutable fixture' and children[0].poll() is not None
    assert m.rt.read_bound(m.rt.reference(case/'pilot_cpu_exit.json'))['reaped'] is True
