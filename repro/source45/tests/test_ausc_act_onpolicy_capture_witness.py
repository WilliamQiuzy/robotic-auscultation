"""Producer/reader CPU integration; these artifacts can never qualify native data."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from test_ausc_act_onpolicy_collection import produced_collection,constructed
from test_ausc_stop_residual_trajectory import native


def stop_tick_setup(tmp_path,monkeypatch,native,*,fault=None):
    """One synthetic cache-recalculation event through the actual capture seam."""
    from types import SimpleNamespace
    from sim.ausc_policy import fresh_expert,m2_stop_residual_trajectory as residual
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    m,c,env,capture=constructed(tmp_path,monkeypatch,lifecycle=True)
    c.target=SimpleNamespace(name=c.metadata['target_id']);c.manager=SimpleNamespace(state=SimpleNamespace(phase=Phase.APPROACH))
    c._l2_started_at=0.;seed=np.full(7,.12345678901234567)
    def leaf(self,command,subphase,**kwargs):
        n=len(env.rows);old=env.value['q'].copy();env.advance();p=c.kin.tcp(command)
        env.value.update(q=command.copy(),dq=(command-old)*240.,tcp_base=p[:3,3],rotation_world=p[:3,:3])
        env.rows[-1].update({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in env.state().items()})
        env.rows[-1].update(q_command=command.tolist(),target=c.target.name,subphase=subphase,physical_violations=[],tcp_speed_m_s=0.)
        c.q=command.copy();capture.after_physics(env,n)
        if subphase=='given_l2_park':
            if fault=='limiter':c.trajectory.input.current_velocity=[.001]*7;c.trajectory.velocity=np.full(7,.001)
            elif fault=='object':c.trajectory.output=deepcopy(c.trajectory.output)
            elif fault=='dq':env.rows[-1]['dq']=[1.]*7
            elif fault=='command':env.rows[-1]['q_command']=[0.]*7
            elif fault=='clock':env.value['time_s']+=1/240.
            elif fault=='two_ticks':env.rows.append(deepcopy(env.rows[-1]))
            elif fault=='event':c._current_stop_residual_audit['event_sha256']='0'*64
        return env.state()
    monkeypatch.setattr(fresh_expert.SkillExpert,'_tick',leaf)
    pose=c.kin.tcp(c.q);env.value.update(tcp_base=pose[:3,3],rotation_world=pose[:3,:3])
    c._tick(seed,'given_initial_settle')
    c.trajectory=residual.StopResidualTrajectory(seed,dt=1/240.,**m.GIVEN_TRAJECTORY_LIMITS)
    transition=m.handoff.apply_stop_profile(c.trajectory,expected_input=m.handoff.input_snapshot(c.trajectory))
    c.l2_handoffs=[dict(parking_stop_profile_transition=transition,parking_preview=dict(stop_residual_adapter={}))]
    c.control_owner='L2_STOP';c._issue_intent(seed,(c.episode_id,c.target.name,0,'APPROACH'),'given_l2_park',stop=True)
    c.trajectory.begin_stop_residual_replanning(history=np.tile(seed,(3,1)))
    raw=c._dispatcher().snapshot(time_s=env.state()['time_s'],physics_row=len(env.rows));command=c.trajectory.step()
    raw['live_target_after_step_q']=c._dispatcher().snapshot(time_s=env.state()['time_s'],physics_row=len(env.rows))['live_target_q']
    c._current_intent_audit=raw;event=c.trajectory.last_stop_residual_event()
    c._current_stop_residual_audit=dict(adapter_id=residual.ADAPTER_ID,event=event,event_sha256=m.handoff.digest(event),
        objects=c.trajectory.last_stop_residual_object_check())
    return m,c,env,capture,command


def test_collection_cache_recalculation_record_size(tmp_path,monkeypatch,native):
    from sim.ausc_policy import expert_capture
    m,c,env,capture,command=stop_tick_setup(tmp_path,monkeypatch,native)
    dumps=json.dumps;sizes=[]
    def measure(value,*args,**kwargs):
        encoded=dumps(value,*args,**kwargs)
        if isinstance(value,dict) and 'interpolator' in value and 'physics_row' in value:sizes.append(len(encoded.encode())+1)
        return encoded
    monkeypatch.setattr(expert_capture.json,'dumps',measure)
    try:c._tick(command,'given_l2_park')
    finally:
        capture.close();print('CPU cache-recalculation record bytes',max(sizes,default=0))
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
    assert rows[-1]['stop_residual_adapter_tick']['event']['cache_recalculated'] is True
    assert sizes and max(sizes)<=65536


@pytest.mark.parametrize('fault',['limiter','object','dq','command','clock','two_ticks','event'])
def test_capture_seam_rejects_actual_mutation_inside_physics(tmp_path,monkeypatch,native,fault):
    _,c,_,capture,command=stop_tick_setup(tmp_path,monkeypatch,native,fault=fault)
    try:
        with pytest.raises(ValueError):c._tick(command,'given_l2_park')
        assert capture._audit_rows==1  # Only the preceding preparation tick persisted.
    finally:capture.close()


@pytest.mark.parametrize('budget',['record','aggregate'])
def test_collection_record_and_aggregate_limits_still_fail_closed(tmp_path,monkeypatch,native,budget):
    from sim.ausc_policy import expert_capture
    _,c,_,capture,command=stop_tick_setup(tmp_path,monkeypatch,native)
    if budget=='record':
        labels=expert_capture.controller_labels
        def padded(*args,**kwargs):return dict(labels(*args,**kwargs),padding='x'*35000)
        monkeypatch.setattr(expert_capture,'controller_labels',padded)
        monkeypatch.setattr(__import__('sim.ausc_policy.fresh_expert',fromlist=['controller_labels']),'controller_labels',padded)
    else:capture.auxiliary_limit=capture._auxiliary_initial_bytes+capture._audit.fileobj.tell()+capture.AUXILIARY_RESERVE_BYTES
    try:
        with pytest.raises(ValueError,match='byte budget'):c._tick(command,'given_l2_park')
        assert capture._audit_rows==1 and capture.errors
    finally:capture.close()


def test_pending_capture_requires_original_token_and_one_tick(tmp_path,monkeypatch,native):
    m,c,env,capture,command=stop_tick_setup(tmp_path,monkeypatch,native)
    from sim.ausc_policy.expert_capture import controller_labels
    labels=controller_labels(c,'given_l2_park',False,False);t=env.state()['time_s']
    evidence=dict(clean_command=dict(q_command=command.tolist(),issued_at_s=t,endpoint_at_s=t+1/240.,readback_at_s=t+1/240.,source_id='unused'))
    try:
        issued=m.capture_evidence(c,evidence)
        with pytest.raises(ValueError,match='Unfinished'):m.capture_evidence(c,evidence)
        with pytest.raises(ValueError,match='Missing/stale'):
            m.finish_stop_capture(c,deepcopy(issued['collection_stop_capture']),rows_before=len(env.rows),
                before=labels,after=labels,clean_command=issued['clean_command'])
        with pytest.raises(ValueError,match='exactly one'):
            m.finish_stop_capture(c,issued['collection_stop_capture'],rows_before=len(env.rows),
                before=labels,after=labels,clean_command=issued['clean_command'])
    finally:capture.close()


@pytest.fixture(scope='module')
def produced(tmp_path_factory):
    # One real writer run supplies all read-only negative cases.
    patch=pytest.MonkeyPatch();root=tmp_path_factory.mktemp('collection-stop-cpu')
    try:
        yield root,patch,produced_collection(root,patch,stop_residual=True,full_cycle=True,native_like=True)
    finally:patch.undo()


def physical_run(value):
    m,c,env,capture,audit,sidecar,records=value
    from sim.ausc_policy import fresh_expert
    plan=m.rt.pose.bound_json(c.reference.snapshot()['sources']['fresh_plan'])
    plane=fresh_expert.contact_plane_contract(plan);phase=dict(precontact_gap_m=.04,precontact_tolerance_m=.002,
        approach_tangential_tolerance_m=.003,angle_tolerance_deg=5.,approach_speed_limit_m_s=.02)
    sources=dict(capture.sources,**{r['path']:r['sha256'] for r in m.sources()['files'].values()})
    sources.update(c.l2_handoffs[-1]['parking_preview']['source_sha256'])
    # Invented CPU environment accounting, never a native owner or physics result.
    return dict(handoff_policy_id=m.handoff.POLICY_ID,runtime_sources_sha256=sources,l2_handoffs=deepcopy(c.l2_handoffs),
        command_collision_samples=len(audit),observed_collision_samples=len(audit),
        posture_command_checks='every commanded physics sample',posture_observation_checks='every observed physics sample',
        phase_failure_diagnostics=[],collision_failure_diagnostics=[],phase_config=phase,
        static_world_verification=dict(checked_physics_steps=len(audit),last_report=dict(clear=True,violations=[]),
            containment_gate=dict(source_npz_sha256=c.metadata['collision_arrays_sha256'])),
        contact_plane_contracts={c.metadata['target_id']:plane},points=[dict(target=c.metadata['target_id'],success=True,
            final_phase='DONE',force_audit=dict(passed=True),contact_plane_audit=dict(passed=True),checked_prepose_gap_m=.03,
            phase_config=dict(phase,precontact_gap_m=.03))])


def test_producer_gzip_passes_unchanged_physical_boundary_and_windows(produced):
    root,_,value=produced;m,c,env,capture,audit,sidecar,records=value
    from sim.ausc_policy import m2_stop_residual_trajectory as residual
    run=physical_run(value)
    residual.verify_native_physical(run,audit,c.metadata,config={residual.CONFIG_KEY:deepcopy(residual.CONFIG)})
    proof=m.verify_captured_boundary(sidecar,audit,records,c.metadata,c.collection_binding)
    boundary,roles=m.capture_roles(sidecar,audit,c.metadata,c.collection_binding)
    samples=m.refinement.teacher_windows(records,audit,c.metadata,boundary_row=boundary,role_proof=roles)
    assert len(samples['states'])>0 and proof['retained_qva_replayed'] and m.cumulative_l2(run,audit)['duration_s']<=120.
    assert c.metadata['source_kind']=='cpu_synthetic_test' and not samples['physical_qualification']
    query=sidecar['intervention']['query'];actual=np.asarray(query['measured_pose_base'])
    assert query['compression_m']<0 and not np.allclose(c.kin.tcp(query['measured_q']),actual,atol=1e-10,rtol=0)
    assert proof['native_tcp_consistency']['position_error_m']>1e-10


@pytest.mark.parametrize('field',['measured_q','measured_pose_base','history','time_s'])
def test_native_roundoff_envelope_does_not_relax_independent_capture_equality(produced,field):
    _,_,value=produced;m,c,_,_,audit,original,records=value
    sidecar=deepcopy(original);intervention=sidecar['intervention'];query=intervention['query']
    candidate=np.asarray(query[field],dtype=float).copy()
    candidate.flat[0]=np.nextafter(candidate.flat[0],np.inf)
    for target in (query,intervention['live_before']['query'],intervention['live_after']['query'],intervention['proposal']['query']):
        target[field]=float(candidate) if candidate.ndim==0 else candidate.tolist()
        if field=='measured_q':target['history'][-1][0]=float(candidate[0])
    intervention['proposal']['query_sha256']=m.rt.digest(query)
    with pytest.raises(ValueError):m.verify_captured_boundary(sidecar,audit,records,c.metadata,c.collection_binding)


@pytest.mark.parametrize('fault',['missing_event','event','digest','objects','owner','offgrid','missing_interpolator',
    'input','output','profile','trajectory','input_object','tail','pose'])
def test_saved_producer_trace_rejects_missing_or_changed_witness(produced,fault):
    _,_,value=produced;m,c,_,_,original,_,_=value
    from sim.ausc_policy import m2_stop_residual_trajectory as residual
    receipt=c.l2_handoffs[-1];start=receipt['parking_start_physics_row']
    # Copy only the parking span; all earlier rows remain read-only inputs.
    audit=original[:start]+deepcopy(original[start:receipt['parking_end_physics_row']])+original[receipt['parking_end_physics_row']:]
    row=audit[start]
    if fault=='missing_event':row.pop('stop_residual_adapter_tick')
    elif fault=='event':row['stop_residual_adapter_tick']['event']['tick']+=1
    elif fault=='digest':row['stop_residual_adapter_tick']['event_sha256']='0'*64
    elif fault=='objects':row['stop_residual_adapter_tick']['objects']['after']['input']+=1
    elif fault=='owner':row['actual_control_owner']='TEACHER_L2'
    elif fault=='offgrid':row['raw_intent']['setter']['offgrid_stop']=False
    elif fault=='missing_interpolator':row.pop('interpolator')
    elif fault=='input':row['interpolator']['input']['fields']['current_velocity'][0]+=.001
    elif fault=='output':row['interpolator']['output']['new_acceleration'][0]+=.001
    elif fault=='profile':row['interpolator']['planning_profile']='l2'
    elif fault=='trajectory':row['interpolator']['trajectory_object']+=1
    elif fault=='input_object':row['interpolator']['objects']['input']+=1
    elif fault=='tail':audit[start+receipt['parking_preview']['stop_steps']]['controller_before']['parking_interpolator_hold']=None
    else:row['parking_pose_after']['pose_base'][0][3]+=.01
    with pytest.raises(ValueError):residual.verify_native_parking(physical_run(value),receipt,audit,c.metadata,c.reference,
        c.manager.config,config={residual.CONFIG_KEY:deepcopy(residual.CONFIG)},kin=c.kin)


def test_public_read_episode_consumes_real_capture_with_cpu_authorization_scaffolding(produced,monkeypatch):
    root,_,value=produced;m,c,env,capture,audit,sidecar,records=value;data=m.data
    from sim.ausc_policy import m2_stop_residual_trajectory as residual
    outer=capture.result(len(audit),execution_completed=True)
    assert outer['compact_capture_complete'] and outer['total_bytes']<=100_000_000
    run=physical_run(value);run.update(raw_multimodal_capture=outer,physics_steps=len(audit),
        executed_controller_class=m.MODULE+'.OnPolicyTeacherCollection',collection_route=m.ROUTE_KEY,
        learner_prefix_executed=True,expert_intervention=True,benchmark_act_eligible=False,learned_policy=True,
        expert_demonstration=False,success=True,physical_validation=True,errors=[],controller_motion_complete=True,
        compact_capture_complete=True,collection_bootstrap=c.bootstrap_ref,cpu_child_exit=sidecar['cpu_child_exit'])
    env.out.mkdir(parents=True,exist_ok=True)
    run['onpolicy_collection']=m.rt.write_new(env.out/'onpolicy_collection.json',sidecar)
    run_ref=m.rt.write_new(root/'COMPLETE.json',run);config=m.rt.write_new(root/'cpu-config.json',c.collection_binding['configuration'])
    child=m.rt.write_new(root/'cpu-child.json',dict(physical_pass=True,identity=dict(pid=12,pgid=11,sid=11,start_ticks='cpu')))
    archive=root/'archive';archive.mkdir();commit=archive/'CODE_COMMIT';commit.write_text(m.sources()['code_commit']+'\n')
    files={};listing=[]
    for i,(original,digest) in enumerate(sorted(run['runtime_sources_sha256'].items())):
        dest=archive/(str(i)+'_'+Path(original).name);dest.write_bytes(Path(original).read_bytes())
        files[original]=m.rt.reference(dest);listing.append(dict(file=dest.name,sha256=digest))
    source_manifest=m.rt.write_new(archive/'SOURCE_MANIFEST.json',listing)
    freeze=m.rt.write_new(root/'cpu-freeze.json',dict(schema=data.SOURCE_FREEZE_SCHEMA,code_commit=m.sources()['code_commit'],
        code_commit_file=m.rt.reference(commit),source_manifest=source_manifest,files=files))
    row=dict(episode_id=c.episode_id,subject_id=c.metadata['subject_id'],target_id=c.metadata['target_id'],split='train',
        pose_id=m.planned_requests()[0]['pose_id'],
        reset_id='cpu-reset',run_id='cpu-owner',job_id='cpu-job',capture_root=str(capture.output),run_result=run_ref,
        config=config,child_result=child,owner_terminal=dict(path=str(root/'cpu-owner.json'),sha256='a'*64),source_freeze=freeze,
        capture_audit=dict(codec='gzip',reference=m.rt.reference(capture.output/'native_audit.jsonl.gz')),
        artifacts={p.relative_to(capture.output).as_posix():dict(bytes=p.stat().st_size,sha256=data.sha256(p))
            for p in capture.output.rglob('*') if p.is_file()})
    attempts=[dict(request=request,episode=row if i==0 else None,outcome='collected' if i==0 else 'unavailable',reason='CPU test only')
        for i,request in enumerate(m.planned_requests())]
    manifest=dict(schema=m.MANIFEST_SCHEMA,prepared=m.PREPARED,initialization={},producer_source=m.sources(),attempts=attempts)
    # Both unmodified entry authorization and the real CPU-source guard reject
    # this fixture; it has neither a native deployment nor live acquisition.
    with pytest.raises(ValueError,match='Explicit collection-only deployment'):m.read_episode(attempts[0],root,manifest)
    with pytest.raises(ValueError,match='Live Isaac'):data._metadata(row,root)
    context=(capture.output,json.loads((capture.output/'compact/manifest.json').read_text()),c.metadata)
    # Only unavailable native authorization/environment leaves are scaffolded.
    # The public body, decoder, gzip/fidelity readers, source freeze, physical,
    # STOP/parking, sidecar/H2/replay, and teacher-window predicates stay real.
    monkeypatch.setattr(m,'preflight',lambda config:c.collection_binding)
    monkeypatch.setattr(data,'_metadata',lambda row,base:context)
    monkeypatch.setattr(data,'_native_evidence',lambda *a,**k:(run,c.collection_binding['configuration'],m.rt.read_bound(freeze)))
    monkeypatch.setattr(data,'_scene_integrity',lambda *a:None)
    samples,receipt,meta=m.read_episode(attempts[0],root,manifest)
    assert len(samples['states'])>0 and receipt['cumulative_l2']['prefix_steps']==9600
    assert meta['source_kind']=='cpu_synthetic_test' and not samples['physical_qualification']


def test_source_catalog_and_actual_payload_sizes(produced):
    _,_,value=produced;m,c,_,capture,audit,_,_=value
    from sim.ausc_policy import m2_act_auxiliary_train as training
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment,m2_rebuild_paired_eval as paired
    refs=m._capture_source_bindings();manifest=json.loads((capture.output/'compact/manifest.json').read_text())
    for name,ref in refs.items():
        assert capture.sources[ref['path']]==manifest['code_bindings'][name]==ref['sha256']
    assert 'm2_given_contact_evaluation.py' in training.code_binding(version=9)['files']
    sizes={name:max((len(json.dumps(row,separators=(',',':'),allow_nan=False).encode())+1 for row in audit if predicate(row)),default=0)
        for name,predicate in dict(ordinary=lambda r:r['actual_control_owner']!='L2_STOP',
            stop=lambda r:'stop_residual_adapter_tick' in r,tail=lambda r:r['actual_control_owner']=='L2_STOP' and
            'stop_residual_adapter_tick' not in r).items()}
    print('CPU producer bytes/catalogs',dict(sizes=sizes,collection=len(m.sources()['files']),
        training=len(training.code_binding(version=9)['files']),deployment=len(deployment.sources(model_schema=training.PILOT_SCHEMA_V9)['files']),
        paired=len(paired.sources(model_schema=training.PILOT_SCHEMA_V9)['files'])))
    assert max(sizes.values())<=65536 and sizes['ordinary']<=16384
    assert all('interpolator' not in row and 'stop_residual_adapter_tick' not in row for row in audit if row['actual_control_owner']!='L2_STOP')
    assert all('native_row' not in row and 'observed_before' not in row and 'observed_after' not in row for row in audit)
