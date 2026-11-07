"""Generated command histories are causal proposals, never measured history."""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest

from test_ausc_given_contact_capture import new_inputs
from test_ausc_given_contact_handoff import FixtureTrajectory
from test_ausc_act_auxiliary_data import pose_inputs,POSE


def module():
    name='sim.ausc_policy.m2_act_planned_supervision'
    assert importlib.util.find_spec(name) is not None, 'Planned supervision module required'
    return importlib.import_module(name)


def test_planned_h2_uses_previous_completed_endpoint_and_raw_future_targets(tmp_path):
    m=module();ref,_,_,meta=new_inputs(tmp_path)
    commands=np.repeat(np.array([[.01]*7,[.02]*7,[.03]*7]),12,axis=0)
    preview=dict(input_state=dict(fields=dict(current_position=[0.]*7)),
        commands=commands.tolist(),cells=[[.4]*7,[.5]*7,[.6]*7],
        all_240hz_commands_verified=True,physics_executed=False)
    arrays,windows=m.planned_windows(preview,meta)
    np.testing.assert_array_equal(arrays['source_histories'][0,:,:7],np.zeros((2,7)))
    np.testing.assert_array_equal(arrays['source_histories'][1,:,:7],[[0.]*7,[.01]*7])
    np.testing.assert_array_equal(arrays['source_times'][:2],[[0.,.05],[.05,.1]])
    np.testing.assert_allclose(arrays['relative_setters'][1,:2],[[.49]*7,[.59]*7])
    assert arrays['setter_is_pad'][1].tolist()==[False,False]+[True]*18
    assert windows[1]['input_index']==1


def test_planned_histories_and_future_labels_do_not_cross_phase_transition(tmp_path,monkeypatch):
    m=module();_,_,_,meta=new_inputs(tmp_path)
    # Explicit encoder injection models a causal 9cm crossing at command endpoint1.
    def encode(q,reference,confidence,metadata):return np.r_[q,[.2 if q[0]==0 else .01,0.,0.],np.zeros(7)]
    monkeypatch.setattr(m.rt.shared.contract,'encode_state',encode)
    preview=dict(input_state=dict(fields=dict(current_position=[0.]*7)),
        commands=np.repeat([[.01]*7,[.02]*7,[.03]*7,[.04]*7],12,axis=0).tolist(),
        cells=[[.4]*7,[.5]*7,[.6]*7,[.7]*7],all_240hz_commands_verified=True,physics_executed=False)
    arrays,windows=m.planned_windows(preview,meta)
    assert [w['input_index'] for w in windows]==[0,2,3]
    assert [w['phase'] for w in windows]==['FREE','APPROACH','APPROACH']
    assert np.count_nonzero(~arrays['setter_is_pad'][0])==1


def scene_fixture(tmp_path):
    m=module();from sim.ausc_policy import m2_given_contact_runtime as rt
    from test_ausc_given_contact_pose import bound
    ref,_,_,meta=pose_inputs(tmp_path)
    value=ref.snapshot();plan=rt.pose.bound_json(value['sources']['fresh_plan'])
    geometry_dir=tmp_path/'geometry';geometry_dir.mkdir()
    (geometry_dir/'collision_geometry.json').write_text('{}')
    with (geometry_dir/'collision_geometry.npz').open('wb') as stream:np.savez(stream,fixture=np.zeros(1))
    artifacts=dict(collision_geometry_sha256=rt.shared.sha256(geometry_dir/'collision_geometry.json'),
        collision_arrays_sha256=rt.shared.sha256(geometry_dir/'collision_geometry.npz'))
    plan.update(transfer_q=[[0.]*7,[0.]*7,[0.]*7],posture_limits={'fixture':True},geometry_artifacts=artifacts)
    value['sources']['fresh_plan']=bound(tmp_path,'planned.json',plan)
    ref=rt.pose.GivenReference.from_snapshot(value);meta=rt.pose.route_metadata(meta,ref)
    config=dict(meta,collision_profile='full',collision_directory=str(geometry_dir),**artifacts,
        patient_receipt=value['sources']['patient_receipt'],world_from_left_base=np.eye(4).tolist(),
        posture_limits=plan['posture_limits'])
    config_ref=rt.write_new(tmp_path/'plan_config.json',config)
    prepared=rt.write_new(tmp_path/'prepared.json',dict(completed=True,physics_executed=False,
        training_eligible=False,source_sha256=rt.geometry.GEOMETRY_SOURCE_SHA256,
        patient=rt.pose.bound_json(value['sources']['patient_receipt'])))
    scene=dict(subject_id=meta['subject_id'],split='train',pose_id=POSE,prepared_complete=prepared,
        patient_receipt=rt.reference(value['sources']['patient_receipt']['file']),
        plans={'L1':dict(plan=rt.reference(value['sources']['fresh_plan']['file']))})
    return scene,meta,config_ref


def test_one_scene_uses_original_preview_and_retains_numerical_rejection(tmp_path):
    m=module();scene,meta,config=scene_fixture(tmp_path)
    seen=[]
    got=m.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,
        verify_factory=lambda *args:lambda q:seen.append(q.copy()),artifact_output=tmp_path/'planned-artifacts')
    assert got['slot']['status']=='accepted' and len(seen)==24+1
    assert all(got['samples'][k] is False for k in ('actions_executed','physics_qualified','native_demonstration_eligible'))
    source=got['samples']['sources'][0]
    assert source['observation_origin']=='causal_generated_20hz_command_states'
    assert source['verification_profile']=='injected_cpu_fixture_unqualified'
    assert source['preview']['input_state']['fields']['current_velocity']==[0.]*7
    def blocked(*args):
        def reject(q):raise ValueError('original collision blocked')
        return reject
    rejected=m.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,verify_factory=blocked,
        artifact_output=tmp_path/'rejected-artifacts')
    assert rejected['slot']['status']=='rejected' and rejected['samples'] is None
    assert 'collision blocked' in rejected['slot']['rejection']


def test_full_raw_setter_artifact_retains_omitted_phase_transition(tmp_path,monkeypatch):
    m=module();scene,meta,config=scene_fixture(tmp_path)
    def encode(q,reference,confidence,metadata):return np.r_[q,[.2 if q[0]==0 else .01,0.,0.],np.zeros(7)]
    monkeypatch.setattr(m.rt.shared.contract,'encode_state',encode)
    raw=np.asarray([[.4]*7,[.5]*7,[.6]*7,[.7]*7])
    def preview(trajectory,*args,**kwargs):
        initial=m.rt.handoff.input_snapshot(trajectory)
        return dict(input_state=initial,input_sha256=m.rt.digest(initial),
            commands=np.repeat([[.01]*7,[.02]*7,[.03]*7,[.04]*7],12,axis=0).tolist(),cells=raw.tolist(),
            all_240hz_commands_verified=True,physics_executed=False,steps=48)
    monkeypatch.setattr(m.recovery,'_preview',preview)
    result=m.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,
        verify_factory=lambda *args:lambda q:None,artifact_output=tmp_path/'planned-artifacts')
    assert result['slot']['status']=='accepted',result['slot']
    source=result['samples']['sources'][0]
    assert [w['input_index'] for w in source['windows']]==[0,2,3]
    full=m.data.load_planned_schedule(source['bindings']['raw_schedule'],{meta['subject_id']})
    np.testing.assert_array_equal(full['absolute_raw_setters'],raw)
    assert full['initial_command_state']==source['preview']['input_state']
    assert source['preview']['raw_setter_cells']==4
    assert all(full[k] is False for k in ('actions_executed','physics_qualified','native_demonstration_eligible'))
    # Full raw labels are bound independently of the window digest.
    arrays={k:result['samples'][k].copy() for k in m.data.ARRAY_KEYS}
    arrays['relative_setters'][0,0,0]+=.001
    with pytest.raises(ValueError,match='schedule'):
        m.data.bind_samples(arrays,[source],{meta['subject_id']})


def test_planned_schedule_rejects_nontrain_before_numeric_read_and_detects_tamper(tmp_path,monkeypatch):
    m=module();scene,meta,config=scene_fixture(tmp_path)
    result=m.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,
        verify_factory=lambda *args:lambda q:None,artifact_output=tmp_path/'planned-artifacts')
    source=result['samples']['sources'][0]
    bound=source['bindings']['raw_schedule'];manifest=m.rt.read_bound(bound)
    bad=deepcopy(manifest);bad['split']='validation'
    bad_ref=m.rt.write_new(tmp_path/'validation-schedule.json',bad)
    with monkeypatch.context() as context:
        context.setattr(np,'load',lambda *args,**kwargs:pytest.fail('Schedule metadata must gate numeric access'))
        with pytest.raises(ValueError,match='TRAIN'):m.data.load_planned_schedule(bad_ref,{meta['subject_id']})
    path=m.rt.bound_path(manifest['arrays'])
    with path.open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='SHA256|hash'):
        m.data.load_planned_schedule(bound,{meta['subject_id']})


def test_trajectory_only_injection_cannot_claim_original_cpu_dynamics(tmp_path,monkeypatch):
    m=module();scene,meta,config=scene_fixture(tmp_path)
    # Use the already validated CPU scene fixture; Mac lacks approved USD/FCL.
    prepared=m._scene(scene,'L1',meta,config,{meta['subject_id']},cpu_fixture=True)
    monkeypatch.setattr(m,'_scene',lambda *args,**kwargs:prepared)
    checked=[]
    monkeypatch.setattr(m,'_original_verifier',lambda *args:lambda q:checked.append(q.copy()))
    result=m.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,
        artifact_output=tmp_path/'trajectory-only-artifacts')
    assert result['slot']['status']=='accepted',result['slot']
    assert len(checked)==25  # Default verifier route still inspects initial + every 240Hz command.
    source=result['samples']['sources'][0]
    assert source['preview']['input_state']['ruckig_version']=='CPU-interface-fixture'
    assert source['verification_profile']=='injected_cpu_fixture_unqualified'
    assert source['injected_factories']==['trajectory_factory']
    assert all(source[key] is False for key in m.data.FLAGS)
