from types import SimpleNamespace as S
import inspect
import numpy as np
import pytest
from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher, collector_source_paths
from sim.auscultation_expert_layout_v1.phase_manager import Phase


@pytest.mark.parametrize('speed',[1.,.8,float('nan')])
def test_fixed_recipe_speed_rejected_before_runtime_construction(speed):
    with pytest.raises(ValueError,match='speed_scale 0.7'):
        ObservedTeacher(speed_scale=speed)


def test_alternate_teacher_feedback_is_rejected():
    with pytest.raises(ValueError,match='measured analytic feedback'):
        ObservedTeacher(action_provider=lambda *args:0.)


def teacher_fixture():
    teacher=ObservedTeacher.__new__(ObservedTeacher)
    teacher.manager=S(state=S(phase=Phase.CONTACT))
    teacher.executor_config=dict(normal_speed_limit_m_s=.004,normal_inward_speed_limit_m_s=.002)
    teacher.speed_scale=.7;teacher.config=S(target_force_N=4.)
    teacher.contact_curves=[dict(nominal_contact_center_gap_m=.005)]
    hall=S(displacement_estimate_m=.003,spring_force_estimate_n=3.,sample_time_s=1.,hall_voltage_v=1.3)
    teacher._tracked_metrics=lambda state:(dict(gap_m=.005,tangent_m=.001,angle_deg=2.),hall,np.eye(4))
    teacher.sensor=S(rows=[dict(index=7,time_s=.95,body_reference_to_current_base=np.eye(4).tolist())])
    teacher.target=S(name='R4');teacher.task_frames={'R4':dict(task_pose_in_body_reference=np.eye(4).tolist())}
    teacher.contact_executor=S(gap=.005,velocity=0.,acceleration=0.)
    teacher._previous_actor_gap=None;teacher._previous_desired=0.
    state=dict(time_s=1.,q=np.zeros(7),dq=np.zeros(7),hall_N=999.,compression_m=999.)
    return teacher,hall,state


def test_force_feedback_and_release_use_adc_despite_opposite_truth():
    teacher,hall,state=teacher_fixture()
    assert teacher.desired_normal_velocity(state)==-.0004
    hall.spring_force_estimate_n=5.
    state['hall_N']=0.;state['compression_m']=0.
    assert teacher.desired_normal_velocity(state)==.0004
    assert not teacher._observed_released(state)
    hall.spring_force_estimate_n=1.5;hall.displacement_estimate_m=0.
    state['hall_N']=999.;state['compression_m']=999.
    assert teacher._observed_released(state)
    assert teacher.desired_normal_velocity(state)==-.0004
    teacher.manager.state.phase=Phase.RETRACT
    assert teacher.desired_normal_velocity(state)==.0002


def test_observation_logs_measured_values_and_geometry_origin():
    teacher,hall,state=teacher_fixture()
    observed,metadata=teacher.observe_actor(state)
    assert metadata['observation_age_s']==pytest.approx(.05)
    result=teacher._skill_observation(observed,metadata)
    assert result['hall_N']==3.
    assert result['compression_m']==.003
    assert result['gap_m']==.005
    assert result['geometry_source_frame']==7
    assert result['observation_source']=='calibrated_ADC_measured_q_registered_RGBD_task'
    assert result['target_position_base_m']==[0.,0.,0.]


@pytest.mark.parametrize('phase,subphase,active,eligible,expected',[
    (Phase.HOLD,'contact_cycle',True,True,'HOLD'),
    (Phase.HOLD,'normal_unload',True,True,'HOLD'),
    (Phase.DONE,'normal_release_confirm',True,True,'RETRACT'),
    (Phase.APPROACH,'transfer',False,True,'FREE'),
    (Phase.APPROACH,'settle',False,False,'FREE'),
    (Phase.DONE,'withdraw_settle',False,False,'FREE'),
])
def test_capture_exactly_once_on_cadence_with_post_update_phase(phase,subphase,active,eligible,expected):
    teacher,_,state=teacher_fixture();teacher.manager.state.phase=phase
    teacher.env=S(rows=[{}]*12);calls=[];teacher.sensor.capture=lambda *args,**kw:calls.append((args,kw))
    teacher._capture_observed_tick(state,subphase,active,False)
    assert len(calls)==1 and calls[0][0][1]==expected and calls[0][1]['eligible']==eligible
    teacher.env.rows.append({})
    teacher._capture_observed_tick(state,subphase,active,False)
    assert len(calls)==1


def test_contact_loop_has_explicit_measured_handoff_and_release_without_truth_reads():
    import ast,textwrap
    method=inspect.getsource(ObservedTeacher._contact_cycle)
    tree=ast.parse(textwrap.dedent(method))
    forbidden=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and
               isinstance(n.value,ast.Name) and n.value.id=='state' and
               isinstance(n.slice,ast.Constant) and n.slice.value in {'hall_N','compression_m'}]
    assert not forbidden
    assert method.index('self._initialize_contact_supervisor()') < method.index('for _ in range')
    assert 'released = self._observed_released(state)' in method


def test_snapshot_includes_actual_v6_supervision_and_timing_sources():
    sources=collector_source_paths()
    for name in ('collect_demo.py','teacher_timing.py','observed_supervisor.py','supervision.py','live_capture.py','render_warmup.py','hall_conditions.py','runtime_dynamics.py','dynamics_contract.py'):
        assert sources['pi05_v6/'+name].is_file()
    assert not any(key.startswith('pi05_v5/') for key in sources)


@pytest.mark.parametrize('measured_loaded', [False, True])
def test_real_collector_loop_brakes_only_after_measured_release(monkeypatch, measured_loaded):
    """Opposed truth/ADC fixtures expose the original hidden release oracle."""
    from sim.auscultation_pi05_v6 import collect_demo as module
    teacher,hall,state=teacher_fixture()
    teacher.manager.state=S(phase=Phase.RETRACT,success=False)
    hall.displacement_estimate_m=.003 if measured_loaded else 0.
    hall.spring_force_estimate_n=4. if measured_loaded else 1.5
    state['hall_N']=0. if measured_loaded else 99.
    state['compression_m']=0. if measured_loaded else 99.
    teacher.kin=S(tcp=lambda q:np.eye(4));teacher.normal=teacher.normal_base=np.array([0.,0.,1.])
    teacher.target.position_base=np.zeros(3);teacher.rotation_base=np.eye(3)
    teacher._point_plan=dict(loaded_q=np.zeros(7),loaded_compression_m=.003)
    teacher.posture_limits={};teacher.q=np.zeros(7);teacher.pre_gap=.01
    teacher._policy_group=0;teacher.control_rows=[]
    teacher.config.max_observation_age_s=.15
    teacher.executor_config.update(nominal_compression_reserve_m=.0005,curve_spacing_m=.00025,
        curve_position_tolerance_m=2e-5,curve_angle_tolerance_rad=2e-4,max_inward_braking_m=1e-6)
    teacher.env=S(base=np.eye(4),rows=[dict(q_command=np.zeros(7))]*3,state=lambda:state)
    events=[]
    teacher._initialize_contact_supervisor=lambda:events.append('handoff')
    curve=S(maximum_gap=.01,report=dict(knots=3,verification_samples=10))
    monkeypatch.setattr(module,'force_derived_domain',lambda *args:dict(nominal_contact_center_gap_m=.005,minimum_gap_m=0.,maximum_nominal_compression_m=.006))
    monkeypatch.setattr(module.FixedNormalCurve,'build',lambda *args,**kwargs:curve)
    executor=S(gap=.009,velocity=.0002,acceleration=0.,stop_pending=False,
               mode='normal',peaks={})
    executor.plan_stop=lambda endpoint:events.append(('stop',endpoint))
    executor.step=lambda desired:np.zeros(7)
    monkeypatch.setattr(module,'NormalExecutor',lambda *args:executor)
    class EndFirstTick(Exception):pass
    def block(command,subphase,**kwargs):
        events.append(('executed_subphase',subphase));raise EndFirstTick()
    teacher._block=block
    with pytest.raises(EndFirstTick):teacher._contact_cycle()
    assert events[0]=='handoff'
    assert (('stop',.01) in events) is not measured_loaded
    assert events[-1]==('executed_subphase','normal_unload' if measured_loaded else 'normal_released_retreat')
    assert teacher._contact_active is False
    assert teacher.contact_curves[-1]['motion_complete'] is False
