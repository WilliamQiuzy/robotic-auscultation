"""Frozen interior trigger, read-only stop preview and actual-pose settling.

No physics, planner, IK, policy target or native safety permission lives here.
Preview clones the actual L2 input, explicitly reserves STOP jerk at 1.49,
then targets the current commanded position. Both complete input snapshots
are bound; the live dispatcher remains its sole owner without a q/v/a reset.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from . import m2_contract as contract
from .m2_act_controller import approach_gate, LIMITS
from .m2_given_contact_pose import need
from sim.auscultation_skill_v1.executor import command_derivatives

POLICY_ID='m2_given_interior_parking_handoff_v3'
POSE_SCHEMA='m2_given_actual_parking_pose_v3'
PREVIEW_SCHEMA='m2_given_current_position_stop_preview_v3'
TAIL_SCHEMA='m2_given_constant_parked_tail_v3'
TAIL_MODE='repeat_verified_stop_command_without_interpolator_step'
DT=1/240.
MAX_PARK_STEPS=1200
SETTLE_INTERVALS=24
MAX_STOP_STEPS=MAX_PARK_STEPS-SETTLE_INTERVALS
MAX_PREVIEW_CPU_S=30.
L2_JERK_NUMERICAL_RESERVE_RAD_S3=1e-4
PLANNING_LIMITS=dict(velocity=.174999,acceleration=.5,jerk=LIMITS['jerk']-L2_JERK_NUMERICAL_RESERVE_RAD_S3)
STOP_PLANNING_LIMITS=dict(PLANNING_LIMITS,jerk=1.49)
TRIGGER_BOUNDS=dict(tangent_m=.002,gap_error_m=.001,full_rotation_rad=float(np.deg2rad(4.)),speed_m_s=.01)
SETTLE_BOUNDS=dict(speed_m_s=.0001,path_m=.00002,rotation_path_rad=float(np.deg2rad(.05)))
INPUT_FIELDS=('control_interface','current_acceleration','current_position','current_velocity',
    'duration_discretization','enabled','intermediate_positions','interrupt_calculation_duration',
    'max_acceleration','max_jerk','max_position','max_velocity','min_acceleration','min_position','min_velocity',
    'minimum_duration','per_dof_control_interface','per_dof_synchronization','per_section_max_acceleration',
    'per_section_max_jerk','per_section_max_position','per_section_max_velocity','per_section_min_acceleration',
    'per_section_min_position','per_section_min_velocity','per_section_minimum_duration','synchronization',
    'target_acceleration','target_position','target_velocity')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pose_metrics(pose, reference):
    pose=contract._pose(pose,'actual handoff pose');delta=pose[:3,3]-reference.pose_base[:3,3]
    gap=float(delta@reference.normal_base)
    return dict(gap_m=gap,tangent_m=float(np.linalg.norm(delta-gap*reference.normal_base)),
        angle_deg=float(np.rad2deg(np.arccos(np.clip(pose[:3,2]@-reference.normal_base,-1.,1.)))))


def interior_trigger(reference, *, command_pose, observed_pose, measured_speed_m_s,
                     phase_config, physics_row, time_s, context, reference_sha256):
    need(reference_sha256==reference.sha256 and len(context)==4 and context[1]==reference.snapshot()['target_id'] and
         context[2]==0 and context[3] in ('FREE','APPROACH'), 'Interior trigger reference/context changed')
    need(type(physics_row) is int and physics_row>=0 and type(time_s) in (int,float) and
         np.isfinite(time_s) and abs(time_s-physics_row*DT)<=1e-8, 'Interior trigger native clock changed')
    speed=contract._scalar(measured_speed_m_s,'actual measured TCP speed')
    need(speed>=0, 'Actual measured speed must be nonnegative')
    result=dict(handoff_policy_id=POLICY_ID,reference_sha256=reference.sha256,reference_revision=0,
        physics_row=physics_row,time_s=float(time_s),context=list(context),bounds=TRIGGER_BOUNDS.copy(),
        fresh_20hz_grid=physics_row%12==0,command_pose_base=contract._pose(command_pose,'command pose').tolist(),
        observed_pose_base=contract._pose(observed_pose,'observed pose').tolist(),measured_speed_m_s=speed)
    passed=result['fresh_20hz_grid'] and context[3]=='APPROACH'
    for name,pose,v in [('command',command_pose,0.),('observed',observed_pose,speed)]:
        metrics=pose_metrics(pose,reference);coarse=approach_gate(dict(metrics,tcp_speed_m_s=v),phase_config)
        angle=float(Rotation.from_matrix(np.asarray(pose)[:3,:3]@reference.pose_base[:3,:3].T).magnitude())
        good=(coarse['passed'] and metrics['tangent_m']<=TRIGGER_BOUNDS['tangent_m'] and
            abs(metrics['gap_m']-phase_config.precontact_gap_m)<=TRIGGER_BOUNDS['gap_error_m'] and
            angle<=TRIGGER_BOUNDS['full_rotation_rad'] and v<=TRIGGER_BOUNDS['speed_m_s'])
        result[name]=dict(coarse=coarse,full_rotation_error_rad=angle,passed=bool(good));passed=passed and good
    result['passed']=bool(passed)
    return result


def measured_sample(state, base, physics_row, speed):
    """Encode env.state actual TCP/full R; never substitute commanded FK."""
    base=contract._pose(base,'world from base');pose=np.eye(4)
    pose[:3,:3]=base[:3,:3].T@contract._array(state.get('rotation_world'),(3,3),'actual native full rotation')
    pose[:3,3]=contract._array(state.get('tcp_base'),(3,),'actual native TCP')
    value=dict(schema=POSE_SCHEMA,source='actual_native_env_state',physics_row=physics_row,
        time_s=contract._scalar(state.get('time_s'),'actual pose time'),
        pose_base=contract._pose(pose,'actual native parking pose').tolist(),
        measured_q=contract._array(state.get('q'),(7,),'actual measured q').tolist(),
        tcp_speed_m_s=contract._scalar(speed,'actual native TCP speed'))
    validate_sample(value)
    return value


def validate_sample(value):
    need(isinstance(value,dict) and value.get('schema')==POSE_SCHEMA and value.get('source')=='actual_native_env_state',
         'Actual full parking pose evidence required')
    row=value.get('physics_row');stamp=value.get('time_s')
    need(type(row) is int and row>=0 and type(stamp) in (int,float) and np.isfinite(stamp) and
         abs(stamp-row*DT)<=1e-8, 'Actual parking pose row/time mismatch')
    pose=contract._pose(value.get('pose_base'),'actual full parking pose')
    contract._array(value.get('measured_q'),(7,),'actual parking measured q')
    speed=contract._scalar(value.get('tcp_speed_m_s'),'actual parking speed')
    need(speed>=0, 'Actual parking speed must be nonnegative')
    return pose,speed


def settle_window(samples):
    need(isinstance(samples,list) and len(samples)<=25, 'At most 25 actual rolling measurements required')
    values=[validate_sample(row) for row in samples]
    need(all(b['physics_row']==a['physics_row']+1 for a,b in zip(samples,samples[1:])),
         'Measured settling history must contain consecutive actual rows')
    path=sum(float(np.linalg.norm(b[0][:3,3]-a[0][:3,3])) for a,b in zip(values,values[1:]))
    angle=sum(float(Rotation.from_matrix(b[0][:3,:3]@a[0][:3,:3].T).magnitude()) for a,b in zip(values,values[1:]))
    speed=max((v for _,v in values),default=0.)
    return dict(handoff_policy_id=POLICY_ID,passed=bool(len(values)==25 and speed<=SETTLE_BOUNDS['speed_m_s'] and
        path<=SETTLE_BOUNDS['path_m'] and angle<=SETTLE_BOUNDS['rotation_path_rad']),samples=len(values),
        intervals=max(0,len(values)-1),physics_rows=[r['physics_row'] for r in samples],
        samples_sha256=digest(samples),max_tcp_speed_m_s=speed,path_length_m=path,rotation_path_rad=angle,
        bounds=SETTLE_BOUNDS.copy())


def _encode(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,(list,tuple)):return [_encode(x) for x in value]
    # pybind Ruckig enums are immutable but not JSON serializable.
    name=str(value).split('.')
    need(len(name)==2 and name[0] in ('ControlInterface','Synchronization','DurationDiscretization'),
         'Unsupported Ruckig input value')
    return dict(enum=name[0],name=name[1])


def input_snapshot(trajectory, *, profile='l2'):
    need(profile in ('l2','stop'), 'Explicit frozen L2 or STOP planning profile required')
    limits=PLANNING_LIMITS if profile=='l2' else STOP_PLANNING_LIMITS
    inp=trajectory.input
    public={k for k in dir(inp) if not k.startswith('_') and not callable(getattr(inp,k))}
    need(public-set(INPUT_FIELDS)<= {'degrees_of_freedom'} and set(INPUT_FIELDS)<=public,
         'Unknown or incomplete actual Ruckig input fields')
    fields={k:_encode(getattr(inp,k)) for k in INPUT_FIELDS}
    need(fields['control_interface'] in ('Position',dict(enum='ControlInterface',name='Position')) and
         fields['synchronization'] in ('Time',dict(enum='Synchronization',name='Time')) and
         fields['duration_discretization'] in ('Continuous',dict(enum='DurationDiscretization',name='Continuous')) and
         fields['enabled']==[True]*7 and fields['intermediate_positions']==[] and
         fields['target_velocity']==fields['target_acceleration']==[0.]*7,
         'Original position-mode current-position stop settings changed')
    optional=set(INPUT_FIELDS)-{'control_interface','synchronization','duration_discretization','enabled',
        'intermediate_positions','current_position','current_velocity','current_acceleration','target_position',
        'target_velocity','target_acceleration','max_velocity','max_acceleration','max_jerk'}
    need(all(fields[k] is None for k in optional), 'Original Ruckig optional settings changed')
    need(trajectory.dt==DT and fields['max_velocity']==[limits['velocity']]*7 and
         fields['max_acceleration']==[limits['acceleration']]*7 and fields['max_jerk']==[limits['jerk']]*7,
         'Frozen actual stop planning limits/DT changed')
    for key in ('current_position','current_velocity','current_acceleration','target_position','target_velocity','target_acceleration'):
        contract._array(fields[key],(7,),'actual Ruckig '+key)
    need(np.array_equal(fields['current_velocity'],trajectory.velocity) and
         np.array_equal(fields['current_acceleration'],trajectory.acceleration), 'Actual Ruckig q/v/a readback differs')
    return dict(dt=trajectory.dt,fields=fields,velocity=np.asarray(trajectory.velocity).tolist(),
        acceleration=np.asarray(trajectory.acceleration).tolist(),ruckig_version=trajectory.module.__version__)


def apply_stop_profile(trajectory, *, expected_input):
    """Only change max_jerk; retain the actual q/v/a and all live objects."""
    before=input_snapshot(trajectory);output=output_snapshot(trajectory)
    need(before==expected_input, 'Actual L2 input changed before explicit STOP profile transition')
    objects=(trajectory,trajectory.input,trajectory.output,trajectory.generator)
    trajectory.input.max_jerk=[STOP_PLANNING_LIMITS['jerk']]*7
    after=input_snapshot(trajectory,profile='stop');expected=deepcopy(before)
    expected['fields']['max_jerk']=[STOP_PLANNING_LIMITS['jerk']]*7
    need(after==expected and output_snapshot(trajectory)==output and
         all(a is b for a,b in zip(objects,(trajectory,trajectory.input,trajectory.output,trajectory.generator))),
         'STOP profile transition changed more than the single jerk bound')
    return dict(schema='m2_given_l2_to_stop_profile_transition_v3',handoff_policy_id=POLICY_ID,
        from_profile=PLANNING_LIMITS.copy(),to_profile=STOP_PLANNING_LIMITS.copy(),changed_fields=['max_jerk'],
        before=before,before_sha256=digest(before),after=after,after_sha256=digest(after))


def _decode(value, module):
    if isinstance(value,list):return [_decode(x,module) for x in value]
    if isinstance(value,dict):
        need(set(value)=={'enum','name'}, 'Unknown serialized Ruckig value')
        return getattr(getattr(module,value['enum']),value['name'])
    return value


def output_snapshot(trajectory):
    output=getattr(trajectory,'output',None)
    if output is None:return None
    return dict(new_position=contract._array(output.new_position,(7,),'live output position').tolist(),
        new_velocity=contract._array(output.new_velocity,(7,),'live output velocity').tolist(),
        new_acceleration=contract._array(output.new_acceleration,(7,),'live output acceleration').tolist(),
        time=contract._scalar(output.time,'live interpolator output time'))


def constant_tail_snapshot(trajectory, command_q, physics_row):
    """Record the completed live interpolator; neither step nor reset it."""
    command=contract._array(command_q,(7,),'verified constant tail command')
    inp=input_snapshot(trajectory,profile='stop');out=output_snapshot(trajectory)
    need(type(physics_row) is int and physics_row>=0 and out is not None and
         np.array_equal(command,inp['fields']['current_position']) and np.array_equal(command,out['new_position']),
         'Constant tail requires the actual completed stop command/input/output')
    objects=dict(trajectory=trajectory,input=trajectory.input,output=trajectory.output,generator=trajectory.generator)
    return dict(schema=TAIL_SCHEMA,mode=TAIL_MODE,handoff_policy_id=POLICY_ID,start_physics_row=physics_row,
        command_q=command.tolist(),input_state=inp,input_sha256=digest(inp),output_state=out,output_sha256=digest(out),
        object_ids={name:id(value) for name,value in objects.items()})


def constant_tail_binding(value):
    need(isinstance(value,dict) and value.get('schema')==TAIL_SCHEMA and value.get('mode')==TAIL_MODE and
         value.get('handoff_policy_id')==POLICY_ID, 'Unknown or missing constant tail mode')
    command=contract._array(value.get('command_q'),(7,),'constant tail command')
    return dict(schema=TAIL_SCHEMA,mode=TAIL_MODE,handoff_policy_id=POLICY_ID,
        start_physics_row=value['start_physics_row'],input_sha256=value['input_sha256'],output_sha256=value['output_sha256'],
        command_sha256=hashlib.sha256(command.tobytes()).hexdigest(),object_ids=deepcopy(value['object_ids']))


def restore_trajectory(snapshot, factory=None, *, profile='l2'):
    if factory is None:
        from sim.auscultation_pi05_v6.execution import JointTrajectory
        factory=JointTrajectory
    need(snapshot.get('dt')==DT and set(snapshot.get('fields',{}))==set(INPUT_FIELDS), 'Exact frozen Ruckig input required')
    need(profile in ('l2','stop'), 'Explicit frozen replay planning profile required')
    clone=factory(snapshot['fields']['current_position'],dt=DT,**(PLANNING_LIMITS if profile=='l2' else STOP_PLANNING_LIMITS))
    for name,value in snapshot['fields'].items():setattr(clone.input,name,_decode(value,clone.module))
    clone.velocity=np.asarray(snapshot['velocity'],float);clone.acceleration=np.asarray(snapshot['acceleration'],float)
    need(input_snapshot(clone,profile=profile)==snapshot, 'Cloned Ruckig input differs from original')
    return clone


@dataclass(frozen=True)
class StopPreview:
    commands: np.ndarray
    report: dict

    def __post_init__(self):
        commands=np.asarray(self.commands,dtype=np.float64)
        object.__setattr__(self,'commands',np.frombuffer(commands.tobytes(),dtype=np.float64).reshape(commands.shape))


def preview_stop(trajectory, history, verify_command, live_snapshot, *, clock=time.monotonic, started_at_s=None):
    start=clock() if started_at_s is None else started_at_s
    def budget():need(clock()-start<MAX_PREVIEW_CPU_S, 'Stop preview CPU deadline')
    budget();before=deepcopy(live_snapshot());original=input_snapshot(trajectory);original_output=output_snapshot(trajectory)
    history=contract._array(history,(3,7),'real stop derivative seam').copy();initial_history=history.copy()
    need(np.array_equal(history[-1],original['fields']['current_position']), 'Stop current q/history seam differs')
    clone=restore_trajectory(original,type(trajectory))
    transition=apply_stop_profile(clone,expected_input=original)
    clone.set_target(np.asarray(original['fields']['current_position']))
    stop_input=input_snapshot(clone,profile='stop')
    commands=[];peaks={k:0. for k in LIMITS}
    try:
        for _ in range(MAX_STOP_STEPS):
            budget();q=contract._array(clone.step(),(7,),'preview native command')
            values=command_derivatives(history,q)
            need(values['velocity']<=LIMITS['velocity'] and values['acceleration']<=LIMITS['acceleration']+1e-8 and
                 values['jerk']<=LIMITS['jerk']+1e-8, 'Stop preview original derivative seam/limits failed')
            for key in peaks:peaks[key]=max(peaks[key],values[key])
            verify_command(q);commands.append(q.copy());history=np.vstack([history[1:],q])
            if (np.max(abs(clone.velocity))<=1e-10 and np.max(abs(clone.acceleration))<=1e-10 and
                    np.array_equal(history,np.tile(q,(3,1)))):break
        else:raise ValueError('Stop preview exceeds 1176 steps reserved inside original 5 second parking budget')
        budget()
    finally:
        after=live_snapshot()
        need(before==after and input_snapshot(trajectory)==original and output_snapshot(trajectory)==original_output,
             'Live input/output/history/physics/reference changed during stop preview')
    commands=np.asarray(commands)
    return StopPreview(commands,dict(schema=PREVIEW_SCHEMA,handoff_policy_id=POLICY_ID,input_state=original,
        input_sha256=digest(original),stop_profile_transition=transition,
        stop_input_state=stop_input,stop_input_sha256=digest(stop_input),
        live_before_sha256=digest(before),live_after_sha256=digest(after),
        live_state_before=before,live_state_after=after,
        frozen_command_history=initial_history.tolist(),target_q=original['fields']['current_position'],
        commands=commands.tolist(),commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
        stop_steps=len(commands),reserved_settle_intervals=SETTLE_INTERVALS,max_parking_steps=MAX_PARK_STEPS,
        l2_planning_limits=PLANNING_LIMITS.copy(),planning_limits=STOP_PLANNING_LIMITS.copy(),
        planning_cpu_s=clock()-start,all_commands_verified=True,
        command_derivative_maxima=peaks,physics_advanced=False,live_state_unchanged=True))
