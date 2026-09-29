"""Causal ADC/FK/RGB-D contact supervision shared by teacher and learned actor.

Current Hall and measured joints update at each real physics tick. Contact
supervision uses the latest available RGB-D body transform, then an optional
camera capture persists exactly one frame with the resulting phase. Geometry
sample provenance is logged; a capture never retroactively changes a decision.
True-physics collision, posture and contact audits remain in the parent executor.
"""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from sim.auscultation_expert_layout_v1.controller import DT, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase, ContactObservation
from .supervision import contact_load_estimate

VERSION = 'observed_adc_fk_rgbd_supervisor_v1'


class ObservedContactSupervisorMixin:
    DECISION_BUFFER_LIMIT=1024

    def __init__(self,*args,sensor,**kwargs):
        if sensor is None:raise ValueError('Observed supervision requires an actual sensor interface')
        self.sensor=sensor
        self.task_frames={}
        self.previous_observed_tcp=None
        self.previous_observed_time=None
        self.observed_tcp_speed=None
        self._last_tcp_measurement=None;self._tcp_speed_reference=None
        self._decision_buffer=[];self._decision_count=0;self._decision_file_initialized=False
        self._observed_base_verification=None
        super().__init__(*args,**kwargs)

    def _verify_observed_base(self,points):
        snapshot=self.env.out/'input_snapshot'
        manifest=json.loads((snapshot/'manifest.json').read_text())
        actual=np.asarray(self.env.base,float)
        if actual.shape!=(4,4) or not np.isfinite(actual).all():
            raise EpisodeRejected('Invalid actual robot base for observed supervision')
        bindings={}
        for target,_ in points:
            binding=manifest['files'][target.name+'/plan']
            digest=binding['sha256']
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise EpisodeRejected('Invalid approved plan source hash')
            path=snapshot/'objects'/digest
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:raise EpisodeRejected('Approved plan source hash mismatch')
            approved=np.asarray(json.loads(raw)['world_from_left_base'],float)
            if approved.shape!=(4,4) or not np.allclose(actual,approved,atol=1e-7,rtol=0):
                raise EpisodeRejected('Actual robot base differs from approved plan')
            bindings[target.name]=digest
        self._observed_base_verification=dict(passed=True,absolute_tolerance=1e-7,plan_sha256=bindings)

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
        hall=self._read_observed_hall(state)
        pose=self.kin.tcp(state['q'],hall.displacement_estimate_m)
        error=pose[:3,3]-point;gap=float(error@normal)
        angle=float(np.rad2deg(np.arccos(np.clip(pose[:3,2]@desired_rotation[:,2],-1,1))))
        return dict(gap_m=gap,tangent_m=float(np.linalg.norm(error-gap*normal)),angle_deg=angle),hall,pose

    def _read_observed_hall(self,state):
        packet=self.sensor.hall.read(state['time_s'])
        hall=self.sensor.cal.decode(packet,now_s=state['time_s'])
        self._last_observed_hall_packet=packet
        return hall

    def _remember_observed_tcp(self,stamp,position):
        """Differentiate consecutive real physics samples, including preparation."""
        position=np.asarray(position,float);stamp=float(stamp)
        if position.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(stamp):
            raise EpisodeRejected('Invalid measured TCP sample')
        if self.previous_observed_time is not None:
            elapsed=stamp-self.previous_observed_time
            if abs(elapsed)<=1e-10:
                if not np.array_equal(position,self.previous_observed_tcp):
                    raise EpisodeRejected('Measured TCP changed without elapsed physical time')
                return self.observed_tcp_speed
            if not np.isclose(elapsed,DT,atol=1e-8,rtol=0):
                raise EpisodeRejected('Measured TCP speed requires consecutive physics samples')
            self.observed_tcp_speed=float(np.linalg.norm(position-self.previous_observed_tcp)/elapsed)
        self.previous_observed_tcp=position.copy();self.previous_observed_time=stamp
        return self.observed_tcp_speed

    def _contact_observation(self,state):
        metric,hall,pose=self._tracked_metrics(state)
        world=self.env.base@pose
        speed=self._remember_observed_tcp(state['time_s'],world[:3,3])
        if speed is None:raise EpisodeRejected('Measured TCP speed unavailable at phase handoff')
        age=max(state['time_s']-self.sensor.rows[-1]['time_s'],state['time_s']-hall.sample_time_s)
        observation=ContactObservation(contact_load_estimate(hall),hall.displacement_estimate_m,
            metric['gap_m'],metric['tangent_m'],metric['angle_deg'],speed,
            observation_age_s=age,tcp_position_m=tuple(world[:3,3]))
        return observation,metric,hall

    def _initialize_contact_supervisor(self):
        # No motion, invented tick, settle credit or teacher command. Evaluate
        # the present sensor sample before choosing the first contact prompt.
        state=self.env.state()
        observation,_,_=self._contact_observation(state)
        previous=self.manager.state
        current=self.manager.update(observation,0.)
        self._record_observed_decision(state,observation,0.,'initialization',previous,current)
        if current.failure_reason:raise EpisodeRejected('Phase failure: '+current.failure_reason)
        return current

    def _record_observed_decision(self,state,observation,dt,kind,previous,current):
        geometry=self.sensor.rows[-1]
        record=dict(schema='wam_observed_supervisor_decision_v1',index=self._decision_count,
            target=self.target.name,physics_row=len(self.env.rows)-1,time_s=float(state['time_s']),
            dt_s=float(dt),decision_kind=kind,measured_q=np.asarray(state['q']).tolist(),
            measured_dq=np.asarray(state['dq']).tolist(),raw_hall_packet=asdict(self._last_observed_hall_packet),
            observation=asdict(observation),geometry=dict(frame_index=geometry['index'],time_s=geometry['time_s'],
                raw_sensor_sha256=geometry['raw_sha256'],
                body_reference_to_current_base=geometry['body_reference_to_current_base']),
            phase_config=asdict(self.manager.config),prior_state=asdict(previous),post_state=asdict(current))
        if self._tcp_speed_reference is None:
            raise EpisodeRejected('Missing actual preceding TCP speed reference')
        record['speed_reference']=self._tcp_speed_reference
        # Validate serialization before counting; the exact used packet/observation
        # is stored, never a later reconstructed ADC sample.
        json.dumps(record,allow_nan=False,separators=(',',':'))
        self._decision_buffer.append(record)
        self._decision_count+=1
        if len(self._decision_buffer)>=self.DECISION_BUFFER_LIMIT:self._flush_observed_decisions()

    def _flush_observed_decisions(self):
        if not self._decision_buffer:return
        path=self.env.out/'supervisor_decisions.jsonl'
        with path.open('a' if self._decision_file_initialized else 'x') as stream:
            stream.writelines(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'
                              for row in self._decision_buffer)
        self._decision_file_initialized=True
        self._decision_buffer.clear()

    def _observed_decision_receipt(self):
        self._flush_observed_decisions()
        path=self.env.out/'supervisor_decisions.jsonl'
        if not self._decision_file_initialized:
            with path.open('x'):pass
            self._decision_file_initialized=True
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        receipt=dict(schema='wam_observed_supervisor_ledger_receipt_v1',
            observed_supervisor_version=VERSION,
            observed_supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ledger_path=path.name,ledger_sha256=digest.hexdigest(),decision_count=self._decision_count,
            buffer_limit=self.DECISION_BUFFER_LIMIT,flushed_on_run_exit=True,
            world_from_left_base=np.asarray(self.env.base,float).tolist(),
            approved_plan_base_verification=self._observed_base_verification,
            scope='actual recorded supervisor inputs and decisions; independent replay audit required')
        (self.env.out/'supervisor_decisions_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    def _capture_observed_tick(self,state,subphase,active,recovery):
        """Learned-runtime capture policy; a collector can override scheduling.

        Called only AFTER any contact phase update. Implementations must persist
        at most one acquired frame for this physics tick and use current manager
        phase for contact supervision, never the prior query's actor_phase.
        """
        actor_phase=getattr(self,'actor_phase',None)
        due=(not self.sensor.rows or state['time_s']-self.sensor.rows[-1]['time_s']>=.05-1e-8)
        if active and due:
            phase=self.manager.state.phase.value
            if phase=='DONE':phase='RETRACT'
            if phase not in {'APPROACH','CONTACT','HOLD','RETRACT'}:
                raise EpisodeRejected('Invalid observed contact phase for capture')
            self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,
                                eligible=not recovery,subphase=subphase)
        elif actor_phase is None and not active and len(self.env.rows)%12==0:
            phase='RETRACT' if self.manager.state.phase in (Phase.RETRACT,Phase.DONE) else self.manager.state.phase.value
            if phase not in {'FREE','APPROACH','CONTACT','HOLD','RETRACT'}:phase='FREE'
            self.sensor.capture(state,phase,self.target.name,len(self.env.rows)-1,
                                eligible=False,subphase=subphase)

    def _tick(self,command,subphase,active=False,recovery=False):
        # Parent retains independent physics/contact/FCL/posture gates. Its
        # ideal-Hall PhaseManager update is deliberately disabled in both users.
        state=super()._tick(command,subphase,active=False,recovery=recovery)
        self.sensor.tick(state)
        hall=self._read_observed_hall(state)
        world=self.env.base@self.kin.tcp(state['q'],hall.displacement_estimate_m)
        self._tcp_speed_reference=self._last_tcp_measurement
        self._remember_observed_tcp(state['time_s'],world[:3,3])
        self._last_tcp_measurement=dict(time_s=float(state['time_s']),physics_row=len(self.env.rows)-1,
            measured_q=np.asarray(state['q']).tolist(),raw_hall_packet=asdict(self._last_observed_hall_packet),
            tcp_position_m=world[:3,3].tolist())
        if active:
            observation,metric,hall=self._contact_observation(state)
            geometry=self.sensor.rows[-1]
            previous=self.manager.state
            current=self.manager.update(observation,DT)
            self._record_observed_decision(state,observation,DT,'physics_tick',previous,current)
            front=self.face_filtered>.15 and self.face_filtered>.1*max(self.total_filtered,.01)
            if previous.phase==Phase.HOLD and current.hold_elapsed>previous.hold_elapsed and front:
                self.audit_hold+=DT
            elif current.phase==Phase.HOLD:self.audit_hold=0.
            self.best_audit_hold=max(self.best_audit_hold,self.audit_hold)
            self.env.rows[-1].update(hold_elapsed_s=current.hold_elapsed,audit_hold_s=self.audit_hold,
                actor_hall_voltage_V=hall.hall_voltage_v,actor_spring_estimate_N=hall.spring_force_estimate_n,
                supervisor_contact_load_estimate_N=observation.hall_force_N,
                actor_gap_m=metric['gap_m'],actor_tangential_m=metric['tangent_m'],actor_angle_deg=metric['angle_deg'],
                observed_supervisor_phase=current.phase.value,
                observed_supervisor_geometry_frame=geometry['index'],
                observed_supervisor_geometry_time_s=geometry['time_s'],
                observed_supervisor_hall_time_s=hall.sample_time_s,
                observed_supervisor_tcp_speed_m_s=observation.tcp_speed_m_s,
                observed_supervisor_observation_age_s=observation.observation_age_s)
        self._capture_observed_tick(state,subphase,active,recovery)
        if self.sensor.rows:self._register_task_frames(self.sensor.rows[0])
        return state

    def run(self,points):
        points=list(points)
        try:
            self._prepare_task_frames(points)
            self._verify_observed_base(points)
            result=super().run(points)
        finally:
            receipt=self._observed_decision_receipt()
        result.update(observed_supervisor_version=VERSION,
            observed_supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            observed_supervisor_order='current ADC/measured q -> latest available RGB-D frame supervision -> post-update camera capture',
            observed_supervisor_speed='consecutive real physics-tick measured-q FK plus decoded Hall; no fabricated initial zero',
            observed_supervisor_ledger=receipt,
            task_frame_registration=self.task_frames)
        return result
