"""Receding-horizon joint policy, never a teacher path/normal projection.

The supplied ten target frames remain task goals and audit references. The
policy receives the measured RGB-D/Hall/joint state, never PhysX contact force.
Preparation and home are explicitly outside the learned task interval.
"""
import json
import time
import numpy as np
from sim.auscultation_skill_v1.controller import ExpertController as SafetyController
from sim.auscultation_expert_layout_v1.controller import DT,CONTROL_STEPS,EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase,ContactObservation
from sim.auscultation_vla_v2.hall import HallPacket
from .execution import JointTrajectory
from .action_contract import compose_joints
from .supervision import contact_load_estimate


class LearnedController(SafetyController):
    def __init__(self,*args,sensor,client,task_scope='full',**kwargs):
        self.sensor=sensor;self.client=client;self.task_scope=task_scope
        self.actor_phase=None;self.learned_rows=[];self.trajectory=None;self.previous_observed_tcp=None
        self.task_frames={}
        super().__init__(*args,**kwargs)

    def _prepare_task_frames(self,points):
        """Copy all supplied initial goals before the scene begins its sequence."""
        if not points or len({target.name for target,_ in points})!=len(points):
            raise ValueError('nonempty unique task sequence required')
        self.initial_task_poses={};self.task_frames={}
        for target,row in points:
            position=np.asarray(target.position_base,float)
            rotation=self.env.base[:3,:3].T@np.asarray(row['command_rotation_world'],float)
            if (position.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(position).all()
                    or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0)
                    or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0)):
                raise ValueError('invalid initial task pose')
            task=np.eye(4);task[:3,3]=position;task[:3,:3]=rotation
            self.initial_task_poses[target.name]=task.copy()

    def _register_task_frames(self,frame):
        """Bind every initial task to one common first observed body transform."""
        if self.task_frames:return
        body=np.asarray(frame['body_reference_to_current_base'],float)
        if body.shape!=(4,4) or not np.isfinite(body).all():raise EpisodeRejected('Invalid measured body transform')
        inverse=np.linalg.inv(body)
        for name,task in self.initial_task_poses.items():
            self.task_frames[name]=dict(initial_task_position=task[:3,3].tolist(),
                initial_task_rotation=task[:3,:3].tolist(),initial_body_reference_to_current_base=body.tolist(),
                task_pose_in_body_reference=(inverse@task).tolist(),source_frame=frame['index'],
                source_time_s=frame['time_s'],registration_observed_target=frame['target'],
                definition='All supplied initial task poses registered together to the first RGB-D body transform; no subsequent simulator refresh')
        (self.env.out/'task_frame_registration.json').write_text(json.dumps(self.task_frames,indent=2)+'\n')

    def _tracked_metrics(self,state):
        """Transport the supplied initial task frame by observed RGB-D motion.

        Initial desired tool orientation is an approved task input: the rim's
        fitted contact angle differs from the surface normal. Preserve that
        explicit fixed task calibration; never refresh it from simulator poses.
        Every initial point is registered against the same first body frame.
        """
        if not self.sensor.rows:raise EpisodeRejected('No initialized body tracker')
        frame=self.sensor.rows[-1]
        if frame['target']!=self.target.name or state['time_s']-frame['time_s']>.15:
            raise EpisodeRejected('Target observation missing or stale')
        body=np.asarray(frame['body_reference_to_current_base'],float)
        if body.shape!=(4,4) or not np.isfinite(body).all():raise EpisodeRejected('Invalid measured body transform')
        self._register_task_frames(self.sensor.rows[0])
        if self.target.name not in self.task_frames:raise EpisodeRejected('Target absent from initial task sequence')
        reference=self.task_frames[self.target.name]
        tracked_task=body@np.asarray(reference['task_pose_in_body_reference'])
        point=tracked_task[:3,3];desired_rotation=tracked_task[:3,:3];normal=-desired_rotation[:,2]
        hall=self.sensor.cal.decode(self.sensor.hall.read(state['time_s']),now_s=state['time_s'])
        pose=self.kin.tcp(state['q'],hall.displacement_estimate_m)
        error=pose[:3,3]-point;gap=float(error@normal)
        angle=float(np.rad2deg(np.arccos(np.clip(pose[:3,2]@desired_rotation[:,2],-1,1))))
        return dict(gap_m=gap,tangent_m=float(np.linalg.norm(error-gap*normal)),angle_deg=angle),hall,pose

    def desired_normal_velocity(self,state):
        raise RuntimeError('No scalar teacher/normal-policy fallback in pose actor')

    def _tick(self,command,subphase,active=False,recovery=False):
        # Keep mesh/posture/force/command gates, disable the inherited phase
        # update that reads ideal simulator Hall. Use calibrated ADC below.
        state=super()._tick(command,subphase,active=False,recovery=recovery)
        self.sensor.tick(state)
        if active:
            if not self.sensor.rows or state['time_s']-self.sensor.rows[-1]['time_s']>=.05-1e-8:
                phase=self.actor_phase or self.manager.state.phase.value
                self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,eligible=True,subphase=subphase)
            metric,hall,pose=self._tracked_metrics(state)
            world=self.env.base@pose;gap=metric['gap_m'];tangent=metric['tangent_m'];angle=metric['angle_deg']
            speed=0. if self.previous_observed_tcp is None else float(np.linalg.norm(world[:3,3]-self.previous_observed_tcp)/DT)
            self.previous_observed_tcp=world[:3,3].copy()
            previous=self.manager.state
            current=self.manager.update(ContactObservation(contact_load_estimate(hall),hall.displacement_estimate_m,
                gap,tangent,angle,speed,tcp_position_m=tuple(world[:3,3])),DT)
            front=self.face_filtered>.15 and self.face_filtered>.1*max(self.total_filtered,.01)
            if previous.phase==Phase.HOLD and current.hold_elapsed>previous.hold_elapsed and front:self.audit_hold+=DT
            elif current.phase==Phase.HOLD:self.audit_hold=0.
            self.best_audit_hold=max(self.best_audit_hold,self.audit_hold)
            self.env.rows[-1].update(hold_elapsed_s=current.hold_elapsed,audit_hold_s=self.audit_hold,
                actor_hall_voltage_V=hall.hall_voltage_v,actor_spring_estimate_N=hall.spring_force_estimate_n,
                supervisor_contact_load_estimate_N=contact_load_estimate(hall),
                actor_gap_m=gap,actor_tangential_m=tangent,actor_angle_deg=angle)
        if self.actor_phase is None and not active and len(self.env.rows)%12==0:
            phase=self.actor_phase or ('RETRACT' if self.manager.state.phase in (Phase.RETRACT,Phase.DONE) else self.manager.state.phase.value)
            if phase not in {'FREE','APPROACH','CONTACT','HOLD','RETRACT'}:phase='FREE'
            eligible=self.actor_phase is not None or active
            self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,eligible=eligible,subphase=subphase)
        if self.sensor.rows:self._register_task_frames(self.sensor.rows[0])
        return state

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
            observation_frame=frame['index'],raw_sensor_sha256=frame['raw_sha256'],
            physics_row_start=start_row,accepted=False,expert_fallback=False)
        self.learned_rows.append(record)
        try:
            raw,receipt=self.client.predict(rgb,frame['state'],phase,self.target.name)
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
            frame=self.sensor.rows[-1]
            distance=float(np.linalg.norm(np.asarray(frame['state'])[14:17]))
            if distance<.09:self.actor_phase='APPROACH'
            metric,_,_=self._tracked_metrics(self.env.state())
            if (self.actor_phase=='APPROACH' and abs(metric['gap_m']-self.pre_gap)<=.002 and
                metric['tangent_m']<=.003 and metric['angle_deg']<=5):break
            self._infer_step(self.actor_phase,active=False)
        else:raise EpisodeRejected('Learned FREE/APPROACH timeout')
        # Decelerate through the same trajectory before contact rather than
        # resetting command velocity/acceleration at a phase boundary.
        self.actor_phase=None
        return canonical

    def _contact_cycle(self):
        self._contact_active=True;self.previous_observed_tcp=None
        if self.trajectory is None:self.trajectory=JointTrajectory(self.q,velocity=self.joint_speed_limit)
        self.actor_phase=self.manager.state.phase.value
        try:
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
        self._prepare_task_frames(points)
        result=super().run(points)
        result.update(controller_source='pi05_measured_joint_policy',task_scope=self.task_scope,
            learned_policy_binding=self.client.binding,learned_policy_queries=self.client.count,
            expert_fallbacks=0,normal_projection=False,force_reference='phase supervisor only; all seven action channels are learned joint offsets',
            phase_supervisor='calibrated Hall ADC and measured-joint FK against RGB-D transported initial task frames',
            realtime_claim=False,simulation_paused_for_inference=True)
        for key in ('actual_contact_policy','contact_execution_method','candidate_timing','candidate_geometry',
                    'candidate_policy','candidate_method','normal_withdrawal_method','normal_withdrawal_scope',
                    'candidate_scope','controller_type','action','route_selection','variation','free_motion_command_bounds'):
            result.pop(key,None)
        result.update(controller_type='pi05_joint_actor_with_independent_safety_supervisor',
            executed_controller_class='sim.auscultation_pi05_v5.learned_controller.LearnedController',
            action='first measured-relative seven-joint proposal; no action channel overwrite',
            motion_execution='continuous Ruckig joint command state; no IK or teacher-curve projection',
            learned_command_limits=dict(joint_velocity_rad_s=self.joint_speed_limit,joint_acceleration_rad_s2=.5,joint_jerk_rad_s3=8.),
            classical_home_limits=dict(joint_velocity_rad_s=.25,joint_acceleration_rad_s2=1.,joint_jerk_rad_s3=1.5),
            task_frame_registration=self.task_frames,
            task_registration='all initial task poses share the first RGB-D body reference; immutable through the sequence',
            inter_point_motion='learned FREE/APPROACH with stationary release confirmation; no classical transfer',
            preparation_home='classical initial preparation and final home explicitly outside learned interval')
        return result
