"""Interior boundary contracts on CPU fixtures, never native qualification."""
from copy import deepcopy
from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_ausc_given_contact_pose import reference
from test_ausc_given_contact_controller import pose_at
from sim.auscultation_expert_layout_v1.phase_manager import PhaseConfig


def module():
    name='sim.ausc_policy.m2_given_contact_handoff'
    assert importlib.util.find_spec(name) is not None, 'Interior handoff helper missing'
    return importlib.import_module(name)


def trigger(tmp_path,**changes):
    ref=reference(tmp_path);p=pose_at(ref,tangent=.001,roll=.01)
    kwargs=dict(command_pose=p,observed_pose=p,measured_speed_m_s=.001,
        phase_config=replace(PhaseConfig(),precontact_gap_m=.03),physics_row=300,time_s=1.25,
        context=('e','L1',0,'APPROACH'),reference_sha256=ref.sha256)
    kwargs.update(changes)
    return module().interior_trigger(ref,**kwargs)


@pytest.mark.parametrize('gap',[.025,.03,.04])
def test_fresh_interior_trigger_preserves_authentic_gap_and_full_rotation(tmp_path,gap):
    ref=reference(tmp_path);p=pose_at(ref,tangent=.001,roll=.01)
    p[:3,3]+=ref.normal_base*(gap-.03)
    result=trigger(tmp_path,command_pose=p,observed_pose=p,phase_config=replace(PhaseConfig(),precontact_gap_m=gap))
    assert result['passed'] and result['handoff_policy_id']=='m2_given_interior_parking_handoff_v3'


@pytest.mark.parametrize('fault',['rim_command','rim_observed','gap','roll','speed','offgrid','stage'])
def test_coarse_entry_is_not_an_interior_stop_trigger(tmp_path,fault):
    ref=reference(tmp_path);p=pose_at(ref,tangent=.001,roll=.01);kw={}
    if fault.startswith('rim'):p[:3,3]+=p[:3,0]*.0015;kw['command_pose' if fault.endswith('command') else 'observed_pose']=p
    elif fault=='gap':p[:3,3]+=ref.normal_base*.0011;kw['observed_pose']=p
    elif fault=='roll':p[:3,:3]=p[:3,:3]@Rotation.from_euler('z',np.deg2rad(4)).as_matrix();kw['observed_pose']=p
    elif fault=='speed':kw['measured_speed_m_s']=.01001
    elif fault=='offgrid':kw.update(physics_row=301,time_s=301/240.)
    else:kw['context']=('e','L1',0,'FREE')
    assert trigger(tmp_path,**kw)['passed'] is False


@pytest.mark.parametrize('kw',[dict(measured_speed_m_s=float('nan')),dict(observed_pose=None),
    dict(reference_sha256='0'*64),dict(context=('e','L1',1,'APPROACH')),dict(time_s=1.3)])
def test_invalid_live_trigger_evidence_refuses(tmp_path,kw):
    with pytest.raises(ValueError):trigger(tmp_path,**kw)


def samples(tmp_path,n=25):
    ref=reference(tmp_path);p=pose_at(ref,tangent=.001,roll=0.)
    return [dict(schema=module().POSE_SCHEMA,physics_row=300+i,time_s=(300+i)/240.,
        pose_base=p.tolist(),measured_q=[0.]*7,tcp_speed_m_s=.00001,
        source='actual_native_env_state') for i in range(n)]


def test_measured_settle_requires_25_real_poses_and_bounds_jitter(tmp_path):
    m=module();history=samples(tmp_path)
    for i,row in enumerate(history):row['pose_base'][0][3]+=(i%2)*.0000004
    result=m.settle_window(history)
    assert result['passed'] and result['samples']==25 and result['intervals']==24
    assert result['path_length_m']<.00002
    assert m.settle_window(history[:-1])['passed'] is False


@pytest.mark.parametrize('fault',['order','duplicate','clock','nonfinite','speed','path','roll','missing'])
def test_measured_settle_rejects_fabricated_or_moving_history(tmp_path,fault):
    m=module();history=samples(tmp_path)
    if fault=='order':history[4],history[5]=history[5],history[4]
    elif fault=='duplicate':history[4]=deepcopy(history[3])
    elif fault=='clock':history[4]['time_s']+=.001
    elif fault=='nonfinite':history[4]['pose_base'][0][3]=float('nan')
    elif fault=='speed':history[4]['tcp_speed_m_s']=.000101
    elif fault=='path':
        for i,row in enumerate(history):row['pose_base'][0][3]+=(i%2)*.000001
    elif fault=='roll':
        for i,row in enumerate(history):
            r=np.asarray(row['pose_base']);r[:3,:3]=r[:3,:3]@Rotation.from_euler('z',(i%2)*np.deg2rad(.003)).as_matrix()
            row['pose_base']=r.tolist()
    else:history[4].pop('pose_base')
    try:result=m.settle_window(history)
    except ValueError:return
    assert result['passed'] is False


class FixtureTrajectory:
    """Deterministic interface fixture; does not substitute for real Ruckig proof."""
    def __init__(self,q,*,dt,velocity,acceleration,jerk):
        m=module();self.dt=dt;self.velocity=np.zeros(7);self.acceleration=np.zeros(7)
        self.input=SimpleNamespace(**{k:None for k in m.INPUT_FIELDS})
        for k,v in dict(current_position=list(q),current_velocity=[0.]*7,current_acceleration=[0.]*7,
            target_position=list(q),target_velocity=[0.]*7,target_acceleration=[0.]*7,
            max_velocity=[velocity]*7,max_acceleration=[acceleration]*7,max_jerk=[jerk]*7,
            enabled=[True]*7,intermediate_positions=[],control_interface='Position',
            synchronization='Time',duration_discretization='Continuous').items():setattr(self.input,k,v)
        self.module=SimpleNamespace(__version__='CPU-interface-fixture')
        self.generator=object()
        self.output=SimpleNamespace(new_position=list(q),new_velocity=[0.]*7,new_acceleration=[0.]*7,time=0.)
    def set_target(self,q):self.input.target_position=np.asarray(q).tolist()
    def step(self):
        q=np.asarray(self.input.current_position).copy();v=np.asarray(self.input.current_velocity)
        if np.max(abs(v))>0:q+=v*self.dt;v=np.zeros(7)
        self.input.current_position=q.tolist();self.input.current_velocity=v.tolist()
        self.velocity=v;self.output.new_position=q.tolist();self.output.new_velocity=v.tolist();self.output.time+=self.dt
        return q


def preview_fixture():
    m=module();t=FixtureTrajectory(np.zeros(7),dt=1/240.,**m.PLANNING_LIMITS)
    t.input.current_velocity=[1e-7]*7;t.velocity=np.asarray(t.input.current_velocity)
    history=np.zeros((3,7));checks=[]
    def state():return dict(input=m.input_snapshot(t),history=history.tolist(),time_s=1.,rows=240,reference='f'*64)
    return t,history,checks,state


def test_stop_profile_changes_only_jerk_without_resetting_state_or_live_objects():
    m=module();t,history,checks,state=preview_fixture()
    objects=(t,t.input,t.output,t.generator);before=m.input_snapshot(t);output=m.output_snapshot(t)
    transition=m.apply_stop_profile(t,expected_input=before)
    assert all(a is b for a,b in zip(objects,(t,t.input,t.output,t.generator)))
    assert m.output_snapshot(t)==output
    after=m.input_snapshot(t,profile='stop');expected=deepcopy(before)
    expected['fields']['max_jerk']=[1.49]*7
    assert after==expected and transition['before']==before and transition['after']==after
    assert transition['changed_fields']==['max_jerk']
    assert transition['before_sha256']==m.digest(before) and transition['after_sha256']==m.digest(after)
    with pytest.raises(ValueError):m.input_snapshot(t)


def test_preview_binds_both_profiles_and_current_position_stop_without_mutating_live():
    m=module();t,history,checks,state=preview_fixture();before=deepcopy(state())
    proposal=m.preview_stop(t,history,lambda q:checks.append(q.copy()),state)
    report=proposal.report
    assert state()==before and report['input_state']==before['input']
    assert report['l2_planning_limits']==dict(velocity=.174999,acceleration=.5,jerk=1.4999)
    assert report['planning_limits']==dict(velocity=.174999,acceleration=.5,jerk=1.49)
    transition=report['stop_profile_transition'];expected=deepcopy(before['input']);expected['fields']['max_jerk']=[1.49]*7
    assert transition['before']==before['input'] and transition['after']==expected
    expected['fields']['target_position']=expected['fields']['current_position']
    assert report['stop_input_state']==expected and report['stop_input_sha256']==m.digest(expected)


@pytest.mark.parametrize('learned',[False,True])
def test_native_stop_applies_only_stop_profile_after_complete_preview(tmp_path,learned):
    c=boundary(tmp_path,learned);c.control_owner='ACT_L2' if learned else 'TEACHER_L2'
    original=deepcopy(module().input_snapshot(c.trajectory));objects=(c.trajectory,c.trajectory.input,c.trajectory.output,c.trajectory.generator)
    assert c._handoff_ready(c._reference_observation())
    c._stop_for_handoff(('e','L1',0,'APPROACH'))
    assert c.trajectory.input.max_jerk==[1.49]*7
    assert all(a is b for a,b in zip(objects,(c.trajectory,c.trajectory.input,c.trajectory.output,c.trajectory.generator)))
    receipt=c.l2_handoffs[-1]
    assert receipt['parking_preview']['stop_profile_transition']['before']==original
    assert receipt['parking_stop_profile_transition']==receipt['parking_preview']['stop_profile_transition']
    assert receipt['parking_tail']['input_state']['fields']['max_jerk']==[1.49]*7


@pytest.mark.parametrize('field,value',[('max_jerk',1.5),('max_jerk',1.48),('max_acceleration',.49),('max_velocity',.174998)])
def test_changed_live_stop_bound_refuses_before_any_native_command(tmp_path,field,value):
    c=boundary(tmp_path);m=module();snapshot=m.input_snapshot(c.trajectory)
    m.apply_stop_profile(c.trajectory,expected_input=snapshot)
    c.control_owner='L2_STOP';c._issue_intent(c.q.copy(),('e','L1',0,'APPROACH'),'given_l2_park',stop=True)
    setattr(c.trajectory.input,field,[value]*7);start=len(c.env.rows)
    with pytest.raises(ValueError,match='planning limits'):c._advance_tick('given_l2_park')
    assert len(c.env.rows)==start


@pytest.mark.parametrize('owner',['TEACHER_L2','ACT_L2'])
def test_stop_profile_cannot_silently_be_reused_for_ordinary_l2(tmp_path,owner):
    c=boundary(tmp_path,learned=owner=='ACT_L2');m=module();c.control_owner=owner
    m.apply_stop_profile(c.trajectory,expected_input=m.input_snapshot(c.trajectory))
    c._issue_intent(c.q.copy(),('e','L1',0,'APPROACH'),'given_l2_motion')
    start=len(c.env.rows)
    with pytest.raises(ValueError,match='planning limits'):c._advance_tick('given_l2_motion')
    assert len(c.env.rows)==start and c.trajectory.input.max_jerk==[1.49]*7


@pytest.mark.parametrize('values',[
    # Four diagnostic f64 scalars only, not a copied raw capture. Original
    # job001 diagnosis SHA eb8e78cd74a36ed9839cea039d3dc37d764f461ec9d842d404a2ee959e1fdfab.
    [1.1276339501647594,1.127641983855574,1.1276492461898124,1.1276556286605295],
    # job004 diagnosis SHA 8d3a07e82b0b87dc620b09dcba504a018615bbad96d6da601bd8d733556022c4.
    [2.1238917525514465,2.123895460095463,2.123898057363376,2.12389965286213],
])
def test_original_quantized_command_jerk_failures_still_refuse_without_epsilon_or_rounding(monkeypatch,values):
    from sim.auscultation_skill_v1.executor import command_derivatives
    m=module();history=np.zeros((3,7));history[:,3]=values[:3];q=np.zeros(7);q[3]=values[-1]
    peaks=command_derivatives(history,q)
    assert peaks['jerk']==1.5000000105374056 and peaks['jerk']>1.5+1e-8
    t=FixtureTrajectory(history[-1],dt=m.DT,**m.PLANNING_LIMITS);before=m.input_snapshot(t)
    monkeypatch.setattr(FixtureTrajectory,'step',lambda self:q.copy())
    def live():return dict(input=m.input_snapshot(t),output=m.output_snapshot(t))
    with pytest.raises(ValueError,match='original derivative'):
        m.preview_stop(t,history,lambda q:pytest.fail('Rejected FD command reached geometric verification'),live)
    assert m.input_snapshot(t)==before and t.input.max_jerk==[1.4999]*7


def test_preview_clones_current_qva_preserves_live_state_and_exact_stop(tmp_path):
    m=module();t,history,checks,state=preview_fixture();before=deepcopy(state())
    proposal=m.preview_stop(t,history,lambda q:checks.append(q.copy()),state)
    assert state()==before and len(checks)==len(proposal.commands)
    assert proposal.report['stop_steps']<=1176 and proposal.report['reserved_settle_intervals']==24
    assert proposal.report['live_before_sha256']==proposal.report['live_after_sha256']
    m.apply_stop_profile(t,expected_input=before['input']);t.set_target(t.input.current_position)
    for command in proposal.commands:np.testing.assert_array_equal(t.step(),command)
    with pytest.raises(ValueError):proposal.commands[0,0]=1.


@pytest.mark.parametrize('fault',['mutation','output_mutation','unsafe','seam','budget','stop_budget','limits','mode','enabled','target_velocity'])
def test_preview_rejects_without_changing_live_target_or_advancing_physics(tmp_path,fault):
    m=module();t,history,checks,state=preview_fixture();before=deepcopy(state());kw={}
    def verify(q):
        if fault=='mutation':t.input.target_position=[1.]*7
        elif fault=='output_mutation':t.output.new_position=[1.]*7
        elif fault=='unsafe':raise ValueError('coarse/posture/collision fixture failure')
    if fault=='seam':history[-2]=1.
    elif fault=='budget':kw.update(clock=lambda:100.,started_at_s=0.)
    elif fault=='stop_budget':
        def moving():t.velocity[:]=.01;return np.asarray(t.input.current_position)
        monkey=pytest.MonkeyPatch();monkey.setattr(FixtureTrajectory,'step',lambda self:(setattr(self,'velocity',np.ones(7)*.01) or np.zeros(7)))
    elif fault=='limits':t.input.max_velocity=[.175]*7
    elif fault=='mode':t.input.control_interface='Velocity'
    elif fault=='enabled':t.input.enabled[0]=False
    elif fault=='target_velocity':t.input.target_velocity=[.001]*7
    try:
        with pytest.raises(ValueError):m.preview_stop(t,history,verify,state,**kw)
    finally:
        if fault=='stop_budget':monkey.undo()
    if fault!='mutation':assert t.input.target_position==before['input']['fields']['target_position']


def test_policy_constants_reserve_settle_inside_single_five_second_interval():
    m=module()
    assert m.POLICY_ID=='m2_given_interior_parking_handoff_v3'
    assert m.MAX_STOP_STEPS+ m.SETTLE_INTERVALS==m.MAX_PARK_STEPS==1200
    assert m.MAX_PREVIEW_CPU_S==30.


def boundary(tmp_path,learned=False):
    from test_ausc_given_contact_controller import bare_boundary,CartesianKin,module as controller
    tmp_path.mkdir(parents=True,exist_ok=True)
    c=bare_boundary(tmp_path);m=controller();h=module()
    if learned:c.__class__=m.GivenContactACT
    c.metadata=__import__('test_ausc_given_contact_pose').metadata(c.reference)
    c.metadata['handoff_policy_id']=h.POLICY_ID;c.episode_id='e';c.target=SimpleNamespace(name='L1')
    c.prefix=None
    c.l2_handoffs=[dict(target='L1',l2_coarse_reached=False)]
    c.manager=SimpleNamespace(config=replace(PhaseConfig(),precontact_gap_m=.03))
    c.env.base=np.eye(4);c.kin=CartesianKin(pose_at(c.reference,tangent=.001,roll=0.))
    c.env.rows=[dict(q_command=[0.]*7,tcp_speed_m_s=0.) for _ in range(240)]
    def state():
        p=c.kin.tcp(c.q)
        return dict(time_s=len(c.env.rows)/240.,q=c.q.copy(),tcp_base=p[:3,3],rotation_world=p[:3,:3],compression_m=0.)
    c.env.state=state;c._deadline=lambda:None
    c._reference_observation=lambda:dict(context=('e','L1',0,'APPROACH'),observed_at_s=c.env.state()['time_s'])
    c.trajectory=FixtureTrajectory(c.q,dt=h.DT,**h.PLANNING_LIMITS)
    c.posture_limits=SimpleNamespace(evaluate=lambda *a,**k:dict(clear=True))
    c.checker=SimpleNamespace(check=lambda *a,**k:dict(clear=True,minimum_distance_m=.01))
    c.static_world=SimpleNamespace(check=lambda:None)
    c._block=lambda commands,label,**kw: [append(c,q) for q in commands]
    c._advance_tick=lambda label:m.GivenContactBoundary._advance_tick(c,label)
    from sim.ausc_policy.m2_act_controller import approach_gate
    def coarse():
        measured=approach_gate(dict(h.pose_metrics(c.kin.tcp(c.q),c.reference),tcp_speed_m_s=0.),c.manager.config)
        return dict(passed=measured['passed'],physics_row=len(c.env.rows),time_s=len(c.env.rows)/240.,
            measured=measured,command_q=c.q.tolist(),measured_q=c.q.tolist())
    c._handoff_measurements=coarse
    c.env.rows[-1]['measurements']=dict(h.pose_metrics(c.kin.tcp(c.q),c.reference),tcp_speed_m_s=0.)
    return c


def append(c,q):
    m=module();before=m.measured_sample(c.env.state(),c.env.base,len(c.env.rows),0.)
    c.q=np.asarray(q).copy();c.env.rows.append(dict(q_command=c.q.tolist(),tcp_speed_m_s=0.))
    if c.control_owner=='L2_STOP':
        after=m.measured_sample(c.env.state(),c.env.base,len(c.env.rows),0.)
        event=c.env.rows[-1];event.update(parking_pose_before=before,parking_pose_after=after,
            physics_row=len(c.env.rows)-1,time_s=len(c.env.rows)/240.,
            controller_after=dict(phase='APPROACH',subphase='given_l2_park'),
            measurements=dict(m.pose_metrics(after['pose_base'],c.reference),tcp_speed_m_s=0.))
        marker=deepcopy(getattr(c,'_parking_tail_audit',None))
        event['controller_before']=dict(phase='APPROACH',subphase='given_l2_park',parking_interpolator_hold=marker)
        event['controller_after']['parking_interpolator_hold']=deepcopy(marker)


def test_earliest_original_coarse_is_immutable_while_genuine_l2_continues(tmp_path):
    c=boundary(tmp_path);p=c.kin.origin.copy();c.kin.origin[:3,3]+=p[:3,0]*.0019
    observation=c._reference_observation()
    assert c._handoff_ready(observation) is False
    first=deepcopy(c.l2_handoffs[0]['coarse_entry']);assert first['physics_row']==240
    assert c.control_owner=='TEACHER_L2' and not getattr(c,'_intent_dispatcher',None)
    c.kin.origin=p;append(c,c.q)
    assert c._handoff_ready(c._reference_observation()) is False  # Actual interior, off-grid 240 Hz row.
    for _ in range(11):append(c,c.q)
    assert c._handoff_ready(c._reference_observation()) is True
    assert c.l2_handoffs[0]['coarse_entry']==first
    assert c.l2_handoffs[0]['parking_trigger']['physics_row']==252


def test_terminal_teacher_waypoint_checks_fresh_interior_before_return(tmp_path):
    from sim.ausc_policy import m2_stage
    c=boundary(tmp_path);c._history=m2_stage.CausalHistory(1.)
    c._history.frames=[(1.,np.zeros(17)),(1.05,np.zeros(17))]
    assert c._teacher_waypoint(c.q.copy()) is True
    assert not getattr(c,'_intent_dispatcher',None)


def test_history_wait_checks_same_trigger_before_any_extra_setter(tmp_path):
    c=boundary(tmp_path)
    assert c._wait_history_cell(c._reference_observation())=='handoff'
    assert not getattr(c,'_intent_dispatcher',None)


def test_changed_episode_context_refuses_before_interior_stop(tmp_path):
    c=boundary(tmp_path);observation=c._reference_observation();observation['context']=('other','L1',0,'APPROACH')
    with pytest.raises(ValueError,match='context'):c._handoff_ready(observation)
    assert not getattr(c,'_intent_dispatcher',None)


@pytest.mark.parametrize('learned',[False,True])
def test_stop_preview_and_measured_settle_are_shared_and_keep_live_interpolator(tmp_path,learned):
    c=boundary(tmp_path,learned);c.control_owner='ACT_L2' if learned else 'TEACHER_L2'
    assert c._handoff_ready(c._reference_observation())
    trajectory=c.trajectory;start=len(c.env.rows)
    result=c._stop_for_handoff(('e','L1',0,'APPROACH'))
    assert result['passed'] and c.trajectory is trajectory and c.control_owner=='L3_ENTRY'
    assert 25<=len(c.env.rows)-start<=1200
    receipt=c.l2_handoffs[0]
    assert receipt['parking_settle']['samples']==25 and receipt['parking_settle']['passed']
    assert receipt['parking_preview']['live_before_sha256']==receipt['parking_preview']['live_after_sha256']


def test_failed_preview_does_not_issue_stop_or_change_actor(tmp_path,monkeypatch):
    c=boundary(tmp_path);assert c._handoff_ready(c._reference_observation())
    monkeypatch.setattr(module(),'preview_stop',lambda *a,**kw:(_ for _ in ()).throw(ValueError('unsafe preview')))
    with pytest.raises(ValueError,match='unsafe preview'):c._stop_for_handoff(('e','L1',0,'APPROACH'))
    assert c.control_owner=='TEACHER_L2' and not getattr(c,'_intent_dispatcher',None)
    assert c.l2_handoffs[-1]['parking_preview_failure']['passed'] is False


def test_constant_tail_does_not_step_or_reset_same_live_interpolator(tmp_path):
    c=boundary(tmp_path);trajectory=c.trajectory;trajectory.generator=object();generator=trajectory.generator
    step=trajectory.step;calls=[]
    def counted():
        calls.append(1);return step()
    trajectory.step=counted
    assert c._handoff_ready(c._reference_observation())
    c._stop_for_handoff(('e','L1',0,'APPROACH'))
    receipt=c.l2_handoffs[-1]
    assert len(calls)==receipt['parking_preview']['stop_steps']
    assert c.trajectory is trajectory and trajectory.generator is generator
    assert receipt['parking_tail']['input_state']==module().input_snapshot(trajectory,profile='stop')
    assert receipt['parking_tail']['output_state']==module().output_snapshot(trajectory)


def test_constant_tail_is_never_a_hold_fallback_for_l2_actions(tmp_path):
    from test_ausc_given_contact_controller import module as controller
    c=boundary(tmp_path);before=len(c.env.rows)
    with pytest.raises(ValueError,match='constant tail'):
        controller().GivenContactBoundary._advance_tick(c,'given_l2_motion',hold_command=c.q.copy())
    assert len(c.env.rows)==before and not getattr(c,'_intent_dispatcher',None)


@pytest.mark.parametrize('calms',[True,False])
def test_rolling_measured_jitter_waits_without_resetting_single_parking_clock(tmp_path,calms):
    c=boundary(tmp_path);original=c.env.state;start=len(c.env.rows)
    def noisy_state():
        value=original();age=len(c.env.rows)-start
        if c.control_owner=='L2_STOP' and (not calms or age<=30):
            value['tcp_base']=value['tcp_base'].copy();value['tcp_base'][0]+=(age%2)*.000003
        return value
    c.env.state=noisy_state;assert c._handoff_ready(c._reference_observation())
    if calms:
        assert c._stop_for_handoff(('e','L1',0,'APPROACH'))['passed']
        assert 30<len(c.env.rows)-start<=1200
    else:
        with pytest.raises(Exception,match='single 5 second'):c._stop_for_handoff(('e','L1',0,'APPROACH'))
        assert len(c.env.rows)-start==1200 and c.control_owner=='L2_STOP'


def test_real_ruckig_preview_live_teacher_student_parity(tmp_path):
    ruckig=pytest.importorskip('ruckig');assert ruckig.__version__=='0.15.3'
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    traces=[];receipts=[];states=[];counts=[];generators=[]
    for learned in (False,True):
        c=boundary(tmp_path/('student' if learned else 'teacher'),learned)
        c.control_owner='ACT_L2' if learned else 'TEACHER_L2'
        t=JointTrajectory(np.zeros(7),dt=module().DT,**module().PLANNING_LIMITS)
        target=np.zeros(7);target[0]=.0001;t.set_target(target);history=[]
        for _ in range(15):history.append(t.step())
        c.trajectory=t;c.q=history[-1].copy()
        generator=id(t.generator);step=t.step;calls=[]
        def counted():calls.append(1);return step()
        t.step=counted
        c.kin.origin[:3,3]-=c.q[:3]
        for row,q in zip(c.env.rows[-3:],history[-3:]):row['q_command']=q.tolist()
        c.env.rows[-1]['measurements']=dict(module().pose_metrics(c.kin.tcp(c.q),c.reference),tcp_speed_m_s=0.)
        assert c._handoff_ready(c._reference_observation())
        before=deepcopy(module().input_snapshot(t));start=len(c.env.rows)
        assert c._stop_for_handoff(('e','L1',0,'APPROACH'))['passed']
        receipt=c.l2_handoffs[-1];assert receipt['parking_preview']['input_state']==before
        assert len(calls)==receipt['parking_preview']['stop_steps']
        assert id(t.generator)==generator and c.trajectory is t
        assert receipt['parking_tail']['input_state']==module().input_snapshot(t,profile='stop')
        assert receipt['parking_tail']['output_state']==module().output_snapshot(t)
        receipts.append(deepcopy(receipt));states.append(module().input_snapshot(t,profile='stop'));counts.append(len(calls));generators.append(generator)
        traces.append(np.asarray([r['q_command'] for r in c.env.rows[start:]]))
    np.testing.assert_array_equal(*traces)
    import os
    output=os.environ.get('AUSC_INTERIOR_PARITY_OUTPUT')
    if output:
        import hashlib,json
        from pathlib import Path
        destination=Path(output);assert destination.is_absolute()
        destination.mkdir(parents=True,exist_ok=False)
        arrays=destination/'canonical_parking.npz'
        def qva(value):
            fields=value['fields'];return np.asarray([fields['current_position'],fields['current_velocity'],fields['current_acceleration']],dtype=np.float64)
        np.savez_compressed(arrays,teacher_parking=traces[0],student_parking=traces[1],
            teacher_preview_stop=np.asarray(receipts[0]['parking_preview']['commands'],dtype=np.float64),
            student_preview_stop=np.asarray(receipts[1]['parking_preview']['commands'],dtype=np.float64),
            teacher_tail_start_qva=qva(receipts[0]['parking_tail']['input_state']),student_tail_start_qva=qva(receipts[1]['parking_tail']['input_state']),
            teacher_tail_end_qva=qva(states[0]),student_tail_end_qva=qva(states[1]))
        result=dict(schema='given_interior_actual_ruckig_parking_cpu_parity_v3',handoff_policy_id=module().POLICY_ID,ruckig_version=ruckig.__version__,
            native_physics=False,physical_qualification=False,training_eligible=False,gpu_work=False,
            feedback='Explicit CPU instantaneous-feedback fixture; not native contact or settling qualification',
            canonical_dtype='float64',array_equal=True,max_abs_teacher_student_delta_rad=float(np.max(abs(traces[0]-traces[1]))),
            live_step_calls=counts,live_generator_identity_unchanged=True,live_generator_ids=generators,
            receipts=receipts,final_input_states=states,
            arrays=dict(file=str(arrays),bytes=arrays.stat().st_size,sha256=hashlib.sha256(arrays.read_bytes()).hexdigest()))
        with (destination/'parity.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')


def test_real_ruckig_finished_same_target_preview_live_parity(tmp_path):
    ruckig=pytest.importorskip('ruckig');assert ruckig.__version__=='0.15.3'
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    for learned in (False,True):
        c=boundary(tmp_path/('student' if learned else 'teacher'),learned)
        c.control_owner='ACT_L2' if learned else 'TEACHER_L2'
        t=JointTrajectory(np.zeros(7),dt=module().DT,**module().PLANNING_LIMITS)
        history=[t.step() for _ in range(15)];pre_clock=t.output.time
        assert np.array_equal(t.input.current_position,t.input.target_position)
        assert t.output.trajectory.duration==0. and t.output.time>0.
        c.trajectory=t;c.q=history[-1].copy();generator=t.generator
        for row,q in zip(c.env.rows[-3:],history[-3:]):row['q_command']=q.tolist()
        assert c._handoff_ready(c._reference_observation())
        result=c._stop_for_handoff(('e','L1',0,'APPROACH'));assert result['passed']
        receipt=c.l2_handoffs[-1]
        assert t.generator is generator and c.trajectory is t
        assert receipt['parking_tail']['input_state']==module().input_snapshot(t,profile='stop')
        assert receipt['parking_tail']['output_state']==module().output_snapshot(t)
        assert receipt['parking_preview']['stop_steps']==1
        assert pre_clock>module().DT and t.output.time==module().DT  # Same target, changed limit must replan.
        from sim.ausc_policy import m2_given_contact_dataset as dataset
        c.metadata.update(episode_id='e',target_id='L1');receipt['parked_entry']=result
        run=dict(handoff_policy_id=module().POLICY_ID,runtime_sources_sha256=receipt['parking_preview']['source_sha256'])
        dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


@pytest.mark.parametrize('job,folder,sha',[
    ('001','given-formal-first100-job001-stop-preview-replay-20261027-v1',
     'eb8e78cd74a36ed9839cea039d3dc37d764f461ec9d842d404a2ee959e1fdfab'),
    ('004','given-formal-first100-job004-stop-preview-replay-20261027-v2',
     '8d3a07e82b0b87dc620b09dcba504a018615bbad96d6da601bd8d733556022c4'),
])
def test_real_ruckig_stop_margin_on_original_failed_diagnostic_input(tmp_path,job,folder,sha):
    """Retained Linux numeric inputs only; full causal/native proof is separate."""
    ruckig=pytest.importorskip('ruckig');assert ruckig.__version__=='0.15.3'
    import hashlib,json,os
    from pathlib import Path
    from sim.auscultation_skill_v1.executor import command_derivatives
    path=Path('/workspace/wam/diagnostics')/folder/'diagnosis.json'
    if not path.exists():pytest.skip('Original immutable Linux diagnostic is required; never mirror raw captures')
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
    diagnostic=json.loads(raw);m=module();inp=diagnostic['reconstructed_stop_input']
    history=np.asarray(diagnostic['pre_stop_native_command_history'],dtype=np.float64)
    assert diagnostic['causal_replay']['all_native_commands_bit_identical'] is True
    assert diagnostic['stop_setter_issued'] is False and diagnostic['physical_qualification'] is False
    assert inp['fields']['max_jerk']==[1.5]*7 and history.shape==(3,7)
    failure=diagnostic['first_derivative_failure']
    assert command_derivatives(np.asarray(failure['history_before_step']),np.asarray(failure['q']))['jerk']>1.5+1e-8
    trajectory=m.restore_trajectory(inp);original=m.input_snapshot(trajectory);objects=(trajectory.input,trajectory.output,trajectory.generator)
    def live():return dict(input=m.input_snapshot(trajectory),output=m.output_snapshot(trajectory),
        command_history=history.tolist(),source_diagnostic_sha256=sha)
    before=deepcopy(live());proposal=m.preview_stop(trajectory,history,lambda q:None,live)
    assert live()==before and m.input_snapshot(trajectory)==original
    transition=m.apply_stop_profile(trajectory,expected_input=original);assert transition==proposal.report['stop_profile_transition']
    trajectory.set_target(history[-1]);assert all(a is b for a,b in zip(objects,(trajectory.input,trajectory.output,trajectory.generator)))
    seam=history.copy();actual=[]
    for expected in proposal.commands:
        q=trajectory.step();np.testing.assert_array_equal(q,expected);actual.append(q.copy())
        peaks=command_derivatives(seam,q)
        assert peaks['velocity']<=.175 and peaks['acceleration']<=.5+1e-8 and peaks['jerk']<=1.5+1e-8
        seam=np.vstack([seam[1:],q])
    assert np.array_equal(seam,np.tile(proposal.commands[-1],(3,1)))
    tail=m.constant_tail_snapshot(trajectory,actual[-1],len(actual));frozen=deepcopy(tail)
    for _ in range(24):assert m.constant_tail_snapshot(trajectory,actual[-1],len(actual))==frozen
    output=os.environ.get('AUSC_STOP_V3_DIAGNOSTIC_OUTPUT')
    if output:
        destination=Path(output)/('job'+job);assert destination.is_absolute();destination.mkdir(parents=True,exist_ok=False)
        arrays=destination/'numeric_stop.npz'
        np.savez_compressed(arrays,preview=proposal.commands,actual=np.asarray(actual),history=history)
        value=dict(schema='given_stop_v3_original_failure_numeric_replay_v1',job=job,ruckig_version=ruckig.__version__,
            handoff_policy_id=m.POLICY_ID,physical_qualification=False,training_eligible=False,physics_executed=False,
            gpu_launched=False,full_causal_generator_replay=False,source_diagnostic=dict(file=str(path),sha256=sha,bytes=len(raw)),
            report=proposal.report,tail=tail,max_abs_delta_rad=float(np.max(abs(np.asarray(actual)-proposal.commands))),
            arrays=dict(file=str(arrays),sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),bytes=arrays.stat().st_size))
        with (destination/'result.json').open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


def reader_fixture(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_dataset as dataset
    c=boundary(tmp_path);c.metadata.update(episode_id='e',target_id='L1')
    assert c._handoff_ready(c._reference_observation())
    result=c._stop_for_handoff(('e','L1',0,'APPROACH'));receipt=c.l2_handoffs[0]
    receipt['parked_entry']=result
    run=dict(handoff_policy_id=module().POLICY_ID,runtime_sources_sha256=receipt['parking_preview']['source_sha256'])
    original=module().restore_trajectory
    monkeypatch.setattr(module(),'restore_trajectory',lambda snapshot,factory=None:original(snapshot,FixtureTrajectory if factory is None else factory))
    monkeypatch.setattr(dataset.shared.contract,'_KIN',c.kin)
    return dataset,c,run,receipt


def test_reader_reconstructs_actual_preview_parity_and_25_measured_poses(tmp_path,monkeypatch):
    dataset,c,run,receipt=reader_fixture(tmp_path,monkeypatch)
    dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


def test_reader_uses_authenticated_actual_trigger_clock_without_rounding(tmp_path,monkeypatch):
    dataset,c,run,receipt=reader_fixture(tmp_path,monkeypatch)
    start=receipt['parking_start_physics_row'];stamp=c.env.rows[start]['parking_pose_before']['time_s']+1.2e-12
    c.env.rows[start]['parking_pose_before']['time_s']=stamp
    preview=receipt['parking_preview']
    for key in ('live_state_before','live_state_after'):preview[key]['measured_pose']['time_s']=stamp
    preview['live_before_sha256']=module().digest(preview['live_state_before'])
    preview['live_after_sha256']=module().digest(preview['live_state_after'])
    receipt['parking_trigger']['time_s']=stamp
    dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)
    receipt['parking_trigger']['time_s']=start/240.
    with pytest.raises(ValueError,match='trigger cannot reconstruct'):
        dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


def test_reader_same_target_changed_limit_uses_replanned_clock_and_rejects_v2_shortcut(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_dataset as dataset
    class ReplanningFixture(FixtureTrajectory):
        """Interface-only cache model; real 0.15.3 test below remains required."""
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.cached=(tuple(self.input.target_position),tuple(self.input.max_jerk))
        def step(self):
            settings=(tuple(self.input.target_position),tuple(self.input.max_jerk))
            if settings!=self.cached:self.output.time=0.;self.cached=settings
            return super().step()
    m=module();c=boundary(tmp_path);c.metadata.update(episode_id='e',target_id='L1')
    t=ReplanningFixture(c.q,dt=m.DT,**m.PLANNING_LIMITS)
    for _ in range(15):t.step()
    original_clock=t.output.time;c.trajectory=t
    assert c._handoff_ready(c._reference_observation())
    result=c._stop_for_handoff(('e','L1',0,'APPROACH'));receipt=c.l2_handoffs[-1];receipt['parked_entry']=result
    assert receipt['parking_preview']['input_state']['fields']['target_position']==receipt['parking_preview']['target_q']
    assert t.output.time==m.DT and original_clock>t.output.time
    run=dict(handoff_policy_id=m.POLICY_ID,runtime_sources_sha256=receipt['parking_preview']['source_sha256'])
    restore=m.restore_trajectory
    monkeypatch.setattr(m,'restore_trajectory',lambda snapshot,factory=None:restore(snapshot,ReplanningFixture if factory is None else factory))
    dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)
    # Re-sign the obsolete target-only retained clock. It must still fail the
    # independent full-input replay even with unchanged real q/v/a and poses.
    tail=receipt['parking_tail'];tail['output_state']['time']=original_clock+m.DT
    tail['output_sha256']=m.digest(tail['output_state'])
    for event in c.env.rows[receipt['parking_stop_complete_physics_row']:]:
        for side in ('controller_before','controller_after'):event[side]['parking_interpolator_hold']=m.constant_tail_binding(tail)
    with pytest.raises(ValueError,match='Completed stop input/output'):
        dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


@pytest.mark.parametrize('fault',['missing','pre_jerk','stop_jerk','extra_changed_field','receipt_transition','stop_target','stop_hash','object_reset','old_policy'])
def test_reader_independently_refuses_resigned_stop_profile_or_object_mutations(tmp_path,monkeypatch,fault):
    dataset,c,run,receipt=reader_fixture(tmp_path,monkeypatch);m=module();preview=receipt['parking_preview']
    if fault=='missing':del preview['stop_profile_transition']
    elif fault in ('pre_jerk','stop_jerk','extra_changed_field'):
        transition=preview['stop_profile_transition'];side='before' if fault=='pre_jerk' else 'after'
        transition[side]['fields']['max_jerk' if fault!='extra_changed_field' else 'max_acceleration']=[1.5 if fault=='stop_jerk' else 1.49]*7
        transition[side+'_sha256']=m.digest(transition[side]);receipt['parking_stop_profile_transition']=deepcopy(transition)
    elif fault=='receipt_transition':receipt['parking_stop_profile_transition']['changed_fields'].append('current_velocity')
    elif fault=='stop_target':
        state=preview['stop_input_state'];state['fields']['target_position'][0]+=.001
        preview['stop_input_sha256']=m.digest(state);receipt['parking_stop_input_state']=deepcopy(state)
        receipt['parking_stop_input_sha256']=m.digest(state)
    elif fault=='stop_hash':receipt['parking_stop_input_sha256']='0'*64
    elif fault=='object_reset':
        before=preview['live_state_before'];before['interpolator_object_ids']['generator']+=1
        preview['live_state_after']=deepcopy(before);preview['live_before_sha256']=preview['live_after_sha256']=m.digest(before)
    else:
        run['handoff_policy_id']=receipt['handoff_policy_id']=c.metadata['handoff_policy_id']='m2_given_interior_parking_handoff_v2'
    with pytest.raises(ValueError):dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


@pytest.mark.parametrize('fault',['missing_tail','unknown_mode','input_mutation','output_mutation','command',
    'start_row','early_tail','missing_tail_tick','input_hash','output_hash','output_clock','object_changed','object_alias'])
def test_reader_independently_refuses_changed_or_premature_constant_tail(tmp_path,monkeypatch,fault):
    dataset,c,run,receipt=reader_fixture(tmp_path,monkeypatch)
    tail=receipt['parking_tail'];start=receipt['parking_start_physics_row'];stop=receipt['parking_stop_complete_physics_row']
    if fault=='missing_tail':del receipt['parking_tail']
    elif fault=='unknown_mode':tail['mode']='continue_ruckig_step'
    elif fault=='input_mutation':tail['input_state']['fields']['current_acceleration'][0]=1e-20
    elif fault=='output_mutation':tail['output_state']['new_velocity'][0]=1e-20
    elif fault=='command':tail['command_q'][0]=1e-20
    elif fault=='start_row':tail['start_physics_row']-=1
    elif fault=='early_tail':c.env.rows[start]['controller_after']['parking_interpolator_hold']=module().constant_tail_binding(tail)
    elif fault=='missing_tail_tick':del c.env.rows[stop]['controller_before']['parking_interpolator_hold']
    elif fault=='input_hash':c.env.rows[stop]['controller_after']['parking_interpolator_hold']['input_sha256']='0'*64
    elif fault=='output_hash':c.env.rows[stop]['controller_after']['parking_interpolator_hold']['output_sha256']='0'*64
    elif fault=='output_clock':
        tail['output_state']['time']+=module().DT;tail['output_sha256']=module().digest(tail['output_state'])
        for event in c.env.rows[stop:]:
            for channel in ('controller_before','controller_after'):
                event[channel]['parking_interpolator_hold']=module().constant_tail_binding(tail)
    elif fault=='object_changed':c.env.rows[stop]['controller_after']['parking_interpolator_hold']['object_ids']['generator']+=1
    else:tail['object_ids']['generator']=tail['object_ids']['trajectory']
    with pytest.raises(ValueError):
        dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)


@pytest.mark.parametrize('fault',['preview_command','native_command','state_hash','input','trigger','history','short',
    'fabricated_pose','roll','clock','budget','sources','stationary_tail','first_reference','first_measurement','first_context'])
def test_reader_refuses_mutated_preview_or_settle_sources(tmp_path,monkeypatch,fault):
    dataset,c,run,receipt=reader_fixture(tmp_path,monkeypatch);preview=receipt['parking_preview'];start=receipt['parking_start_physics_row']
    if fault=='preview_command':preview['commands'][0][0]+=.001
    elif fault=='native_command':c.env.rows[start]['q_command'][0]+=.001
    elif fault=='state_hash':preview['live_after_sha256']='0'*64
    elif fault=='input':preview['input_state']['fields']['current_velocity'][0]=.1
    elif fault=='trigger':receipt['parking_trigger']['physics_row']+=1
    elif fault=='history':preview['frozen_command_history'][0][0]=.1
    elif fault=='short':receipt['parking_settle']['samples']=24
    elif fault=='fabricated_pose':c.env.rows[start]['parking_pose_before']['source']='command_fk'
    elif fault=='roll':
        p=np.asarray(c.env.rows[-1]['parking_pose_after']['pose_base']);p[:3,:3]=p[:3,:3]@Rotation.from_euler('z',.01).as_matrix()
        c.env.rows[-1]['parking_pose_after']['pose_base']=p.tolist()
    elif fault=='clock':c.env.rows[-1]['parking_pose_after']['time_s']+=.001
    elif fault=='budget':receipt['parking_steps']=1201
    elif fault=='sources':run['runtime_sources_sha256']={}
    elif fault=='stationary_tail':c.env.rows[-1]['q_command'][0]+=.001
    elif fault=='first_reference':receipt['coarse_entry']['reference_sha256']='0'*64
    elif fault=='first_measurement':receipt['coarse_entry']['measured']['tangent_m']=.003
    else:receipt['coarse_entry']['context'][0]='wrong-episode'
    with pytest.raises(ValueError):dataset._interior_parking(run,receipt,c.env.rows,c.metadata,c.reference,c.manager.config,kin=c.kin)
