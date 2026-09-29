"""Receding-horizon joint policy, never a teacher path/normal projection.

The supplied ten target frames remain task goals and audit references. The
policy receives the measured RGB-D/Hall/joint state plus the explicitly supplied
task pose transported by that RGB-D frame, never PhysX contact force.
Preparation and home are explicitly outside the learned task interval.
"""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from sim.auscultation_skill_v1.controller import ExpertController as SafetyController
from sim.auscultation_expert_layout_v1.controller import DT,CONTROL_STEPS,EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from .execution import JointTrajectory
from .action_contract import compose_joints
from .supervision import contact_load_estimate
from . import phase_contract
from .observed_supervisor import ObservedContactSupervisorMixin
from .task_observation import encode_task_state, observation_binding
from sim.auscultation_vla_v2.hall import HallPacket


class LearnedController(ObservedContactSupervisorMixin,SafetyController):
    def __init__(self,*args,sensor,client,task_scope='full',**kwargs):
        if client.binding.get('observation_contract')!=observation_binding():
            raise ValueError('Policy observation contract differs from runtime task adapter')
        self.client=client;self.task_scope=task_scope
        self.actor_phase=None;self.learned_rows=[];self.trajectory=None
        super().__init__(*args,sensor=sensor,**kwargs)




    def desired_normal_velocity(self,state):
        raise RuntimeError('No scalar teacher/normal-policy fallback in pose actor')





    def _infer_step(self,phase,*,active):
        state=self.env.state();stamp=float(state['time_s'])
        if (not self.sensor.rows or abs(self.sensor.rows[-1]['time_s']-stamp)>1e-8 or
            self.sensor.rows[-1]['target']!=self.target.name):
            self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,eligible=True,subphase='learned_observation')
        frame=self.sensor.rows[-1]
        if frame['target']!=self.target.name:raise EpisodeRejected('Policy observation target mismatch')
        self._tracked_metrics(state) # Require the common registration before learned motion.
        with np.load(self.sensor.out/frame['raw_file'],allow_pickle=False) as a:rgb=a['rgb']
        start_row=len(self.env.rows)
        record=dict(index=len(self.learned_rows),phase=phase,target=self.target.name,time_s=stamp,
            observation_contract=observation_binding(),
            acquired_observation_phase=frame['phase'],requested_policy_phase=phase,
            phase_decision_provenance='shared observed selector/supervisor at query time; acquired frame phase is immutable acquisition metadata',
            observation_frame=frame['index'],raw_sensor_sha256=frame['raw_sha256'],
            physics_row_start=start_row,accepted=False,expert_fallback=False)
        self.learned_rows.append(record)
        try:
            if self.client.binding.get('observation_contract')!=record['observation_contract']:
                raise EpisodeRejected('Policy observation contract changed before query')
            policy_state=encode_task_state(frame['state'],measured_q=frame['q'],hall_packet=HallPacket(**frame['hall']),
                observation_time_s=frame['time_s'],now_s=stamp,
                body_reference_to_current_base=frame['body_reference_to_current_base'],body_time_s=frame['time_s'],
                task_pose_in_body_reference=self.task_frames[self.target.name]['task_pose_in_body_reference'],
                kinematics=self.kin,hall_calibration=self.sensor.cal)
            record['policy_state']=policy_state.tolist()
            raw,receipt=self.client.predict(rgb,policy_state,phase,self.target.name)
            record.update(raw_actions=raw.tolist(),inference=receipt,
                executed_chunk_prefix=1,
                force_semantics='supervisor-owned phase reference; no extra force-feedback motion')
            if len(self.env.rows)!=start_row or self.env.state()['time_s']!=stamp:
                raise EpisodeRejected('Physics advanced during policy inference')
            goal=compose_joints(state['q'],raw[0],phase,self.kin.lower_limits,self.kin.upper_limits)
            record.update(measured_q=state['q'].tolist(),goal_q=goal.tolist())
            if self.trajectory is None:self.trajectory=JointTrajectory(self.q,velocity=self.joint_speed_limit)
            self.trajectory.set_target(goal)
            record['accepted']=True
            previous_phase=self.manager.state.phase
            previous_revision=frame['target_estimate']['revision']
            for _ in range(CONTROL_STEPS):
                command=self.trajectory.step()
                self._block(command[None],'learned_'+phase.lower(),active=active)
                if self.manager.state.failure_reason:raise EpisodeRejected('Phase failure: '+self.manager.state.failure_reason)
                if active and self.manager.state.phase!=previous_phase:
                    record['prefix_interrupted_for_phase_change']=True;break
                if self.sensor.rows[-1]['target_estimate']['revision']!=previous_revision:
                    record['prefix_interrupted_for_target_revision']=True;break
        except Exception as e:
            record['failure_reason']=type(e).__name__+': '+str(e);raise
        finally:
            record['physics_row_end_exclusive']=len(self.env.rows)
            with (self.env.out/'learned_actions.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')

    def _move_to_prepose(self,row):
        self._point_plan=row
        if self.task_scope=='contact':
            # Explicit component test, never reported as full learned-route success.
            return super()._move_to_prepose(row)
        canonical,index=self._canonical(row);self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        if self.sequence_index==0:
            if np.max(abs(self.q))>.03:raise EpisodeRejected('Expected actual initial home')
            self._hold(1.,'settle')
        self.trajectory=JointTrajectory(self.q,velocity=self.joint_speed_limit)
        self.actor_phase='FREE'
        state=self.env.state()
        if not self.sensor.rows or self.sensor.rows[-1]['target']!=self.target.name:
            self.sensor.capture(state,'FREE',self.target.name,len(self.env.rows)-1,eligible=True,subphase='new_target_observation')
        for _ in range(2400):
            state=self.env.state()
            if abs(self.sensor.rows[-1]['time_s']-state['time_s'])>1e-8:
                self.sensor.capture(state,self.actor_phase,self.target.name,len(self.env.rows)-1,
                                    eligible=True,subphase='learned_observation')
            frame=self.sensor.rows[-1]
            self.actor_phase=phase_contract.precontact_phase(self.actor_phase,frame['state'])
            if self.actor_phase=='APPROACH' and self._initialize_contact_supervisor().phase==Phase.CONTACT:break
            self._infer_step(self.actor_phase,active=False)
        else:raise EpisodeRejected('Learned FREE/APPROACH timeout')
        # Retain command velocity/acceleration across the measured phase handoff.
        self.actor_phase=None
        return canonical

    def _contact_cycle(self):
        self._contact_active=True
        if self.trajectory is None:self.trajectory=JointTrajectory(self.q,velocity=self.joint_speed_limit)
        try:
            self._initialize_contact_supervisor()
            for _ in range(2400):
                phase=self.manager.state.phase
                if self.manager.state.failure_reason:raise EpisodeRejected('Phase failure: '+self.manager.state.failure_reason)
                if phase==Phase.ABORT:raise EpisodeRejected('Learned contact aborted')
                self.actor_phase='RETRACT' if phase==Phase.DONE else phase.value
                state=self.env.state();metric,hall,_=self._tracked_metrics(state)
                if phase==Phase.DONE and metric['gap_m']>=self.pre_gap-.002 and contact_load_estimate(hall)<.2:
                    # Reach a stationary retained command through Ruckig, not a q jump.
                    self.trajectory.set_target(self.q)
                    for _ in range(120):
                        command=self.trajectory.step();self._block(command[None],'learned_release_stop',active=True)
                        if np.max(abs(self.trajectory.velocity))<1e-10 and np.max(abs(self.trajectory.acceleration))<1e-10:break
                    else:raise EpisodeRejected('Learned release stop timeout')
                    if self.best_audit_hold<self.config.hold_duration_s-2*DT:
                        raise EpisodeRejected('Independently verified continuous hold missing')
                    self._normal_release_target=self.target.name
                    return
                self._infer_step(self.actor_phase,active=True)
            raise EpisodeRejected('Learned contact/release timeout')
        finally:self._contact_active=False;self.actor_phase=None

    def run(self,points):
        points=list(points)
        if self.task_scope=='contact' and len(points)>1:
            raise ValueError('contact component scope supports only one point')
        result=super().run(points)
        result.update(controller_source='pi05_absolute_joint_policy',task_scope=self.task_scope,
            observation_contract=observation_binding(),
            phase_contract_version=phase_contract.VERSION,
            phase_contract_sha256=hashlib.sha256(Path(phase_contract.__file__).read_bytes()).hexdigest(),
            learned_policy_binding=self.client.binding,learned_policy_queries=self.client.count,
            expert_fallbacks=0,normal_projection=False,force_reference='phase supervisor only; all seven action channels are learned absolute joint targets',
            phase_supervisor='calibrated Hall ADC and measured-joint FK against RGB-D transported initial task frames',
            realtime_claim=False,simulation_paused_for_inference=True)
        for key in ('actual_contact_policy','contact_execution_method','candidate_timing','candidate_geometry',
                    'candidate_policy','candidate_method','normal_withdrawal_method','normal_withdrawal_scope',
                    'candidate_scope','controller_type','action','route_selection','variation','free_motion_command_bounds'):
            result.pop(key,None)
        result.update(controller_type='pi05_joint_actor_with_independent_safety_supervisor',
            executed_controller_class='sim.auscultation_pi05_v6.learned_controller.LearnedController',
            action='first absolute seven-joint command target; no action channel overwrite',
            motion_execution='continuous Ruckig joint command state; no IK or teacher-curve projection',
            learned_command_limits=dict(joint_velocity_rad_s=self.joint_speed_limit,joint_acceleration_rad_s2=.5,joint_jerk_rad_s3=8.),
            classical_home_limits=dict(joint_velocity_rad_s=self.joint_speed_limit,joint_acceleration_rad_s2=1.,joint_jerk_rad_s3=1.5),
            task_frame_registration=self.task_frames,
            task_registration='all initial task poses share the first RGB-D body reference; immutable through the sequence',
            inter_point_motion='learned FREE/APPROACH with stationary release confirmation; no classical transfer',
            preparation_home='classical initial preparation and final home explicitly outside learned interval')
        return result
