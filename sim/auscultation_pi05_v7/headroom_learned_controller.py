"""Independent runtime identity: original learned behavior, internal jerk headroom.

The three copied methods differ only in their trajectory factory. Tests compare
normalized ASTs against the frozen V6 implementation. The entire original
observed supervisor / safety _tick chain is inherited without an override.
"""
import json
import numpy as np
from sim.auscultation_pi05_v6.learned_controller import LearnedController
from sim.auscultation_expert_layout_v1.controller import DT,CONTROL_STEPS,EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from sim.auscultation_pi05_v6.action_contract import compose_joints
from sim.auscultation_pi05_v6.supervision import contact_load_estimate
from sim.auscultation_pi05_v6 import phase_contract
from sim.auscultation_pi05_v6.task_observation import encode_task_state,observation_binding
from sim.auscultation_vla_v2.hall import HallPacket
from sim.auscultation_pi05_v6.dynamics_contract import execution_contract,sha256_json
from .headroom_execution import HeadroomJointTrajectory,runtime_adaptation_contract,RUNTIME_JERK

CLASS='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
SOURCE='pi05_absolute_joint_policy_with_runtime_jerk_headroom'


class HeadroomLearnedController(LearnedController):
    def __init__(self,*args,**kwargs):
        if kwargs.get('task_scope','full')!='full':raise ValueError('Headroom adaptation supports full learned task only')
        super().__init__(*args,**kwargs)
        if self.joint_speed_limit!=.175 or self.speed_scale!=.7:
            raise ValueError('Headroom adaptation requires original gentle timing')

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
            if self.trajectory is None:self.trajectory=HeadroomJointTrajectory(self.q,velocity=self.joint_speed_limit)
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
        self.trajectory=HeadroomJointTrajectory(self.q,velocity=self.joint_speed_limit)
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
        if self.trajectory is None:self.trajectory=HeadroomJointTrajectory(self.q,velocity=self.joint_speed_limit)
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
        result=super().run(points)
        descriptor=runtime_adaptation_contract()
        result.update(executed_controller_class=CLASS,controller_source=SOURCE,
            runtime_controller_generation='pi05_v7_runtime_jerk_headroom_over_pi05_v6',
            trained_command_limits=dict(joint_velocity_rad_s=.175,joint_acceleration_rad_s2=.5,joint_jerk_rad_s3=8.),
            learned_command_limits=dict(joint_velocity_rad_s=.175,joint_acceleration_rad_s2=.5,joint_jerk_rad_s3=RUNTIME_JERK),
            runtime_adaptation_contract=descriptor,runtime_adaptation_contract_sha256=sha256_json(descriptor),
            trained_execution_sha256=descriptor['trained_execution_sha256'],
            runtime_execution_sha256=descriptor['runtime_execution_sha256'],
            checkpoint_retrained_for_runtime=False,original_failed_case_reclassified=False)
        return result
