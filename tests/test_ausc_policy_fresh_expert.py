import hashlib
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import fresh_expert as f
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256


def test_preload_is_physical_slider_effort_and_never_a_force_label():
    indices, efforts = f.passive_effort(11, 1.5)
    assert indices.tolist() == [11]
    assert efforts.tolist() == [-1.5]
    for bad in (0., 2., np.nan, True):
        with pytest.raises(ValueError):
            f.passive_effort(11, bad)


def fixture_plan(tmp_path):
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.ausc_policy.geometry_epoch import contact_frame
    kin = OpenArmKinematics(contact_frame=contact_frame)
    q = np.array([0., -1., .2, 1.4, .1, .1, -.1])
    pose = kin.tcp(q, .0025)
    normal = -pose[:3, 2]
    target = pose[:3, 3]-normal*2e-5
    pre = kin.solve_ik(target+normal*.03, orientation_base=pose[:3, :3], seed=q,
                       max_starts=1)
    assert pre.success
    limits = PostureLimits(10., tuple(kin.lower_limits), tuple(kin.upper_limits)).to_dict()
    geometry = tmp_path/'geometry'; geometry.mkdir()
    hashes = {}
    for name, key in [('collision_geometry.json', 'collision_geometry_sha256'),
                      ('collision_geometry.npz', 'collision_arrays_sha256')]:
        data = ('current '+name).encode(); (geometry/name).write_bytes(data)
        hashes[key] = hashlib.sha256(data).hexdigest()
    row = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
               freshly_generated=True, geometry_artifacts=hashes, name='R3', success=True,
               world_from_left_base=np.eye(4).tolist(), position_world=target.tolist(),
               normal_world=normal.tolist(), command_normal_world=normal.tolist(),
               command_rotation_world=pose[:3,:3].tolist(), expected_contact_center_world=pose[:3,3].tolist(),
               expected_contact_gap_m=2e-5, free_approach_start_gap_m=.03,
               transfer_q=[np.zeros(7).tolist(), (pre.q*.5).tolist(), pre.q.tolist()],
               lower_staging_q=(pre.q*.5).tolist(), approach_q=[pre.q.tolist(), pre.q.tolist()],
               loaded_q=q.tolist(), loaded_compression_m=.0025, posture_limits=limits,
               geometric_clearance_m=.003, loaded_noncontact_clearance_m=.0003,
               joint_edge_sample_step_rad=.015)
    row['gap_axis']='command_normal'
    row['selected_fit']=dict(gap_axis='command_normal',first_contact=dict(
        gap_axis='command_normal',passed=True,raw_anchor_world_m=target.tolist(),
        center_world_m=pose[:3,3].tolist(),first_contact_search_resolution_m=1e-6,
        contact_surface_gap_bracket_m=[0.,1e-9],diaphragm_residual_distance_m=2e-5,
        fit=dict(passed=True,diaphragm_collision=False,
                 distances_m=dict(fixed=.01,clamp=.005,shell=.00032,diaphragm=2e-5))))
    path=tmp_path/'plan.json'; path.write_text(json.dumps(row))
    binding=dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config=dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                freshly_generated=True, posture_limits=deepcopy(limits),
                fresh_expert=dict(plans=[binding], passive_preload_N=1.5))
    return row, config, geometry, kin


def test_hashbound_new_plan_keeps_surface_point_and_tcp(tmp_path):
    row, config, geometry, kin = fixture_plan(tmp_path)
    points, metadata = f.read_plans(config, geometry, kin)
    target, actual = points[0]
    np.testing.assert_array_equal(target.position_world, row['position_world'])
    assert actual['expected_contact_gap_m'] == 2e-5
    assert metadata['plans'][0]['sha256'] == config['fresh_expert']['plans'][0]['sha256']
    path = config['fresh_expert']['plans'][0]['file']
    with open(path, 'a') as stream: stream.write(' ')
    with pytest.raises(ValueError, match='plan.*changed'):
        f.read_plans(config, geometry, kin)


@pytest.mark.parametrize('mutation',['missing','coarse','different_line','shell_clearance'])
def test_coarse_or_mismatched_contact_fit_cannot_enter_runtime(tmp_path,mutation):
    from pathlib import Path
    row,config,geometry,kin=fixture_plan(tmp_path)
    receipt=row['selected_fit']['first_contact']
    if mutation=='missing':row.pop('selected_fit')
    elif mutation=='coarse':receipt['diaphragm_residual_distance_m']=.0015
    elif mutation=='different_line':row['gap_axis']='surface_normal'
    else:receipt['fit']['distances_m']['shell']=.000299
    path=Path(config['fresh_expert']['plans'][0]['file']);path.write_text(json.dumps(row))
    config['fresh_expert']['plans'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='contact|clearances'):
        f.read_plans(config,geometry,kin)


@pytest.mark.parametrize('mutation', ['epoch', 'arrays', 'pose', 'posture', 'home'])
def test_retired_or_disconnected_plans_refused_before_isaac(tmp_path, mutation):
    row, config, geometry, kin = fixture_plan(tmp_path)
    if mutation == 'epoch': row['geometry_id'] = 'legacy_1262'
    elif mutation == 'arrays': row['geometry_artifacts']['collision_arrays_sha256'] = '0'*64
    elif mutation == 'pose': row['expected_contact_center_world'][0] += .0058
    elif mutation == 'posture': row['posture_limits']['max_elbow_z_m'] -= .01
    elif mutation == 'home': row['transfer_q'][0][0] = .1
    path = config['fresh_expert']['plans'][0]['file']
    from pathlib import Path
    Path(path).write_text(json.dumps(row))
    config['fresh_expert']['plans'][0]['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with pytest.raises(ValueError): f.read_plans(config, geometry, kin)


def test_free_path_checks_every_command_before_first_execution():
    controller = f.FreshExpert.__new__(f.FreshExpert)
    controller.q = np.zeros(7)
    controller.env = SimpleNamespace(rows=[])
    controller.target = SimpleNamespace(name='R3')
    controller.free_motion_timing=[]; controller.free_collision_samples=0
    queried=[]; executed=[]
    def check(q, **kwargs):
        queried.append(q.copy())
        assert not executed
        return dict(clear=True, minimum_distance_m=.003)
    controller.checker = SimpleNamespace(check=check)
    controller._block=lambda commands, label: executed.extend(commands)
    controller._continuous_path(np.array([np.zeros(7), np.full(7, .02)]), 'transfer')
    assert len(queried) == len(executed) > 100
    report = controller.free_motion_timing[0]
    assert report['motion_complete'] and report['peaks']['jerk_rad_s3'] <= 1.5+1e-9


def test_a_late_collision_prevents_the_entire_free_motion():
    controller = f.FreshExpert.__new__(f.FreshExpert)
    controller.q=np.zeros(7);controller.env=SimpleNamespace(rows=[])
    controller.target=SimpleNamespace(name='R3');controller.free_motion_timing=[]
    controller.free_collision_samples=0
    executed=[]
    controller.checker=SimpleNamespace(check=lambda q, **kw: dict(
        clear=bool(q[0]<.019), minimum_distance_m=.003 if q[0]<.019 else .001))
    controller._block=lambda commands,label: executed.extend(commands)
    with pytest.raises(Exception, match='free command'):
        controller._continuous_path(np.array([np.zeros(7), np.full(7,.02)]), 'transfer')
    assert not executed


def test_physical_audit_is_per_point_and_never_training_acceptance():
    def row(target, force):
        return dict(target=target, diaphragm_N=force, lip_N=0., contact_rim_N=0.,
                    carriage_other_N=0., frame_N=0., forbidden_N=0., compression_m=.0025,
                    hall_N=4., training_accepted=True)
    rows=[row('R3',4.) for _ in range(720)]+[row('L3',0.) for _ in range(720)]
    outcome=dict(success=True, training_accepted=True, points=[
        dict(target='R3', success=True, start_physics_row=0, end_physics_row=720),
        dict(target='L3', success=True, start_physics_row=720, end_physics_row=1440)])
    result=f.audit_physics(outcome, rows)
    assert not result['physical_validation']
    assert result['points'][0]['force_audit']['passed']
    assert not result['points'][1]['force_audit']['passed']
    assert not result['training_eligible']
    assert not any(r['training_accepted'] for r in rows)
    assert not result['training_accepted']


def test_any_nonmembrane_contact_immediately_rejects(monkeypatch):
    monkeypatch.setattr(f.SkillExpert,'_violations',lambda self,state:([],4.))
    controller=f.FreshExpert.__new__(f.FreshExpert)
    for group in f.FORBIDDEN:
        controller.env=SimpleNamespace(last_contacts={group:1e-12})
        violations,_=controller._violations({})
        assert any(group in reason for reason in violations)
    controller.env=SimpleNamespace(last_contacts={'diaphragm':4.})
    assert controller._violations({})[0]==[]


def test_missing_actual_force_cannot_become_physical_success():
    rows=[dict(target='R3',hall_N=4.) for _ in range(720)]
    outcome=dict(success=True,points=[dict(target='R3',success=True,
        start_physics_row=0,end_physics_row=720)])
    result=f.audit_physics(outcome,rows)
    assert not result['success'] and not result['physical_validation']
    assert result['controller_motion_complete']


@pytest.mark.parametrize('fault', ['save', 'nonfinite_control', 'diagnostics', 'video'])
def test_cleanup_failure_still_closes_video_and_stops_world(tmp_path, fault):
    called=[]
    def save(name):
        called.append('save')
        if fault=='save': raise OSError('trace disk failure')
    def close():
        called.append('video_close')
        if fault=='video': raise OSError('encoder failed')
        return dict(file='actual_fresh_expert.mp4')
    env=SimpleNamespace(rows=[dict(training_accepted=True)],video_recorder=object(),
        save=save,world=SimpleNamespace(stop=lambda:called.append('world_stop')))
    controller=SimpleNamespace(control_rows=[dict(value=float('nan') if fault=='nonfinite_control' else 1.)],
        contact_curves=[dict(value=float('nan'))] if fault=='diagnostics' else [],
        free_motion_timing=[],inter_point_routes=[])
    video,errors=f.finalize_probe(env,controller,SimpleNamespace(close=close),tmp_path)
    assert called==['save','video_close','world_stop']
    assert env.video_recorder is None and not env.rows[0]['training_accepted']
    assert errors and all('stage' in error and 'error' in error for error in errors)
    if fault!='video': assert video['file']=='actual_fresh_expert.mp4'


def test_cleanup_accumulates_failures_instead_of_short_circuiting(tmp_path):
    called=[]
    def fail(stage):
        called.append(stage)
        raise OSError(stage)
    env=SimpleNamespace(rows=[],video_recorder=None,save=lambda name:fail('save'),
        world=SimpleNamespace(stop=lambda:fail('stop')))
    controller=SimpleNamespace(control_rows=[dict(value=float('nan'))],contact_curves=[],
                               free_motion_timing=[],inter_point_routes=[])
    video,errors=f.finalize_probe(env,controller,SimpleNamespace(close=lambda:fail('video')),tmp_path)
    assert video is None
    assert called==['save','video','stop']
    assert [error['stage'] for error in errors]==['trace_save','control_save','video_finalize','world_stop']


def test_run_probe_revokes_a_passing_physics_result_on_trace_save_failure(tmp_path,monkeypatch):
    import sys
    from sim.ausc_policy import contact_collision
    called=[]
    rows=[dict(target='R3',diaphragm_N=4.,lip_N=0.,contact_rim_N=0.,carriage_other_N=0.,
        frame_N=0.,forbidden_N=0.,compression_m=.0025,hall_N=4.) for _ in range(720)]
    class Env:
        def __init__(self,output,**kwargs):
            self.rows=rows;self.base=np.eye(4);self.out=output
            self.world=SimpleNamespace(stop=lambda:called.append('stop'))
        def state(self):return dict(q=np.zeros(7))
        def save(self,name):
            called.append('save')
            raise OSError('disk write failed')
    class Controller:
        def __init__(self,*args,**kwargs):self.control_rows=[]
        def run(self,points):
            return dict(success=True,points=[dict(target='R3',success=True,
                start_physics_row=0,end_physics_row=720)])
    class Recorder:
        def __init__(self,*args):pass
        def close(self):called.append('close');return dict(file='preserved.mp4')
    metadata=dict(geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        freshly_generated=True,geometry_artifacts={},world_from_left_base=np.eye(4).tolist(),plans=[])
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(f,'read_plans',lambda *args:([object()],metadata))
    monkeypatch.setattr(contact_collision,'ContactCollisionChecker',lambda *args:SimpleNamespace(static_bundle={}))
    monkeypatch.setattr(f,'FreshContactEnv',Env);monkeypatch.setattr(f,'FreshExpert',Controller)
    monkeypatch.setattr(f,'_VideoRecorder',Recorder)
    monkeypatch.setattr(f,'require_geometry_artifacts',lambda *args:{})
    config=dict(geometry_source_sha256=GEOMETRY_SOURCE_SHA256,fresh_expert=dict(passive_preload_N=1.5))
    result=f.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert called==['save','close','stop']
    assert result['controller_motion_complete']
    assert result['points'][0]['force_audit']['passed']
    assert not result['success'] and not result['physical_validation']
    assert result['video']['file']=='preserved.mp4'
    assert any(error['stage']=='trace_save' for error in result['errors'])
    saved=json.loads((tmp_path/'fresh_expert_result.json').read_text())
    assert not saved['success'] and saved['errors']==result['errors']


def test_video_release_is_independent_of_encoder_log_close(tmp_path):
    called=[]
    def stderr_close():
        called.append('stderr_close');raise OSError('log stream failure')
    recorder=f._VideoRecorder.__new__(f._VideoRecorder)
    recorder.output=tmp_path;recorder.frames=[dict(video_frame=0,physics_row=23)]
    recorder.process=SimpleNamespace(stdin=SimpleNamespace(close=lambda:called.append('stdin_close')),
        wait=lambda **kwargs:0)
    recorder.stderr=SimpleNamespace(close=stderr_close)
    recorder.annotator=SimpleNamespace(detach=lambda:called.append('detach'))
    recorder.product=SimpleNamespace(destroy=lambda:called.append('destroy'))
    with pytest.raises(RuntimeError,match='stderr_close'):recorder.close()
    assert called==['stdin_close','stderr_close','detach','destroy']
    assert json.loads((tmp_path/'video_frames.json').read_text())==recorder.frames


def test_contact_timing_rejects_the_real_40mm_failure_before_any_motion(tmp_path,monkeypatch):
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    row,_,_,kin=fixture_plan(tmp_path)
    target=SimpleNamespace(name='R3',position_world=np.asarray(row['position_world']))
    controller=f.FreshExpert.__new__(f.FreshExpert)
    controller.kin=kin;controller.speed_scale=.7
    controller.config=PhaseConfig(contact_timeout_s=20/.7)
    controller.executor_config=dict(normal_inward_speed_limit_m_s=.002)
    controller.contact_preflights=[]
    called=[]
    monkeypatch.setattr(f.SkillExpert,'run',lambda self,points:called.append('motion'))
    short=deepcopy(row);short['free_approach_start_gap_m']=.025
    long=deepcopy(row);long['free_approach_start_gap_m']=.04
    with pytest.raises(f.EpisodeRejected,match='contact timing necessary lower bound'):
        controller.run([(target,short),(target,long)])
    assert not called
    report=controller.contact_preflights[-1]
    assert report['minimum_travel_s']==pytest.approx((.04-2e-5)/.0014)
    assert report['necessary_lower_bound_s']==pytest.approx((.04-2e-5)/.0014+.3)
    assert report['contact_timeout_s']==pytest.approx(20/.7)
    assert not report['passed'] and not report['success_guaranteed']


def test_contact_timing_uses_loaded_surface_projection_and_keeps_timeout(tmp_path):
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig
    row,_,_,kin=fixture_plan(tmp_path)
    target=SimpleNamespace(name='R3',position_world=np.asarray(row['position_world']))
    config=PhaseConfig(contact_timeout_s=20/.7)
    # The actual finite head's first-contact center can be outside the target
    # point plane. Time is measured to its FK projection, not to an assumed 0.
    row['free_approach_start_gap_m']=.04
    target.position_world-=np.asarray(row['command_normal_world'])*.002
    report=f.contact_timing_preflight(kin,target,row,config,.0014)
    assert report['surface_gap_m']==pytest.approx(.00202)
    assert report['minimum_travel_s']==pytest.approx((.038-2e-5)/.0014)
    assert report['passed'] and not report['success_guaranteed']
    assert config.contact_timeout_s==20/.7
    row['free_approach_start_gap_m']=.025
    target.position_world=np.asarray(row['position_world'])
    report=f.contact_timing_preflight(kin,target,row,config,.0014)
    assert report['passed']
    assert report['necessary_lower_bound_s']==pytest.approx((.025-2e-5)/.0014+.3)


def test_original_phase_timeout_stops_before_executor_step_and_is_preserved(monkeypatch,tmp_path):
    from sim.auscultation_expert_layout_v1.phase_manager import PhaseManager,PhaseConfig,ContactObservation
    from sim.auscultation_skill_v1 import controller as skill
    cfg=PhaseConfig(precontact_gap_m=.04,contact_timeout_s=20/.7)
    manager=PhaseManager(cfg)
    def obs(gap):return ContactObservation(0.,0.,gap,0.,0.,.0014)
    manager.update(obs(.04),1/240)
    for _ in range(6858): manager.update(obs(.0003479074491176248),1/240)
    assert manager.state.failure_reason=='contact_timeout'
    controller=f.FreshExpert.__new__(f.FreshExpert)
    controller.config=cfg;controller.manager=manager
    controller.executor_config={**skill.CONTACT_DEFAULTS,**skill.config_values(),
                                'joint_velocity_limit_rad_s':.175}
    controller.speed_scale=.7;controller.pre_gap=.04
    controller.sensor_condition=controller.action_provider=None
    controller.contact_curves=[];controller.phase_failure_diagnostics=[]
    controller._point_plan=dict(loaded_q=np.zeros(7),loaded_compression_m=.0025)
    controller.q=np.zeros(7);controller.normal=controller.normal_base=np.array([0.,0.,1.])
    controller.rotation_base=np.eye(3);controller.posture_limits=None
    controller.target=SimpleNamespace(name='R3',position_base=np.zeros(3))
    controller.kin=SimpleNamespace(tcp=lambda q:np.eye(4))
    controller._skill_observation=lambda state,metadata:{}
    controller.metrics=lambda state:dict(gap_m=.0003479074491176248)
    state=dict(time_s=80.47916666667564,hall_N=0.,compression_m=0.)
    controller.env=SimpleNamespace(base=np.eye(4),rows=[dict(q_command=np.zeros(7)) for _ in range(3)],
                                   state=lambda:state)
    controller._policy_group=0
    class Executor:
        def __init__(self,*args):
            self.gap=.00033625000000172064;self.velocity=-.0014;self.acceleration=0.
        def step(self,desired):raise AssertionError('executor must not move after original phase failure')
    curve=SimpleNamespace(report=dict(knots=3,verification_samples=9),maximum_gap=.04)
    monkeypatch.setattr(skill.FixedNormalCurve,'build',lambda *a,**kw:curve)
    monkeypatch.setattr(skill,'NormalExecutor',Executor)
    with pytest.raises(f.EpisodeRejected,match='contact_timeout'):
        controller._contact_cycle()
    report=controller.phase_failure_diagnostics[-1]
    assert report['phase']=='RETRACT' and report['phase_failure_reason']=='contact_timeout'
    assert report['observed_gap_m']==pytest.approx(.0003479074491176248)
    assert report['executor_gap_m']==pytest.approx(.00033625000000172064)
    assert report['executor_velocity_m_s']==-.0014 and report['executor_acceleration_m_s2']==0.
    assert report['rejected_before_next_executor_step']
    assert len(controller.env.rows)==3
    # The same evidence is preserved even if the run exits through an exception.
    controller.control_rows=[];controller.free_motion_timing=[];controller.inter_point_routes=[]
    env=SimpleNamespace(rows=[],video_recorder=None,save=lambda name:None,world=SimpleNamespace(stop=lambda:None))
    _,errors=f.finalize_probe(env,controller,None,tmp_path)
    assert not errors
    saved=json.loads((tmp_path/'fresh_expert_controller_diagnostics.json').read_text())
    assert saved['phase_failure_diagnostics']==controller.phase_failure_diagnostics


def test_successful_hold_retraction_keeps_the_original_velocity_policy():
    from sim.auscultation_expert_layout_v1.phase_manager import Phase
    controller=f.FreshExpert.__new__(f.FreshExpert)
    controller.manager=SimpleNamespace(state=SimpleNamespace(phase=Phase.RETRACT,failure_reason=None))
    controller.action_provider=None;controller.speed_scale=.7
    controller.executor_config=dict(normal_speed_limit_m_s=.004,normal_inward_speed_limit_m_s=.002)
    assert controller.desired_normal_velocity(dict(hall_N=4.,compression_m=.0025))==.004
