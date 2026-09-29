"""Genuine nonzero demonstration starts over the unchanged V6 contact teacher.

Preparation is real, logged motion. No joint reset, synthetic frame, home washout,
contact fallback, altered force law, or qualification is performed here.
"""
import hashlib
from pathlib import Path
import time
import numpy as np
from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher
from sim.auscultation_expert_layout_v1.controller import EpisodeRejected, FRONT_GROUPS, FORBIDDEN_GROUPS
from sim.expert_quality_20260924.candidate_planner import checked_edge, joint_path

CLASS='sim.auscultation_pi05_v7.start_teacher.StartObservedTeacher'
SOURCE='analytic_expert_shared_measured_supervisor_with_bound_start'
RECEIPT_SCHEMA='wam_v7_measured_start_receipt_v1'


def checked_first_state(state,requested):
    q=np.asarray(state['q'],float);dq=np.asarray(state['dq'],float);requested=np.asarray(requested,float)
    if (q.shape!=(7,) or dq.shape!=(7,) or requested.shape!=(7,)
            or not np.isfinite(np.r_[q,dq,requested,state['time_s']]).all()):
        raise ValueError('Nonfinite or malformed measured start')
    if np.max(abs(q-requested))>.03 or np.max(abs(q))<=.03:
        raise EpisodeRejected('First eligible measured start misses request or washed out to home')
    return q,dq


def join_actual_route(retained,measured,bridge,canonical,stage_index):
    bridge=joint_path(bridge);canonical=joint_path(canonical)
    if (not np.allclose(bridge[0],measured,atol=1e-9,rtol=0)
            or not np.allclose(bridge[-1],canonical[stage_index],atol=1e-9,rtol=0)):
        raise ValueError('Actual measured bridge endpoints differ')
    route=np.vstack([bridge,canonical[stage_index+1:]]);index=len(bridge)-1
    if np.max(abs(np.asarray(retained)-measured))>1e-9:
        route=np.vstack([retained,route]);index+=1
    if np.any(np.max(abs(route),axis=1)<=.03):
        raise ValueError('Eligible route returns to zero home')
    return route,index


class StartObservedTeacher(ObservedTeacher):
    def __init__(self,*args,start_manifest,start_manifest_sha256,experiment_protocol_sha256,
                 episode_id,**kwargs):
        self.start_manifest=start_manifest
        self.start_manifest_sha256=start_manifest_sha256
        self.experiment_protocol_sha256=experiment_protocol_sha256
        self.episode_id=episode_id;self.start_role=None
        self.start_receipt=dict(schema=RECEIPT_SCHEMA,episode_id=episode_id,
            target=start_manifest['target'],start_family=start_manifest['start_family'],
            experiment_protocol_sha256=experiment_protocol_sha256,
            start_manifest_sha256=start_manifest_sha256,
            source_plan_sha256=start_manifest['parent_plan_sha256'],
            requested_start_q_rad=start_manifest['requested_start_q_rad'],
            preparation_physics_range=None,settle_physics_range=None,first_eligible=None,
            executed_controller_class=CLASS,teacher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification_passed=False,dataset_sealed=False,
            scope='Producer observation only; independent raw replay and full qualification still required')
        super().__init__(*args,**kwargs)

    def _capture_observed_tick(self,state,subphase,active,recovery):
        # Add an explanatory field to the just-executed row; never rewrite its
        # subphase, q/dq, phase, timestamp, or capture schedule.
        if self.start_role is not None and not recovery:
            self.env.rows[-1]['preparation_role']=self.start_role
        before=len(self.sensor.rows)
        super()._capture_observed_tick(state,subphase,active,recovery)
        if len(self.sensor.rows)==before:return
        frame=self.sensor.rows[-1]
        if frame['eligible'] and self.start_receipt['first_eligible'] is None:
            if self.start_role!='start_observation' or subphase!='transfer' or active or recovery:
                raise EpisodeRejected('First eligible frame escaped declared start observation')
            q,dq=checked_first_state(state,self.start_manifest['requested_start_q_rad'])
            row=len(self.env.rows)-1
            if (frame['physics_row']!=row or frame['phase']!='FREE' or frame['subphase']!=subphase
                    or frame['time_s']!=state['time_s'] or row%12!=11
                    or not np.array_equal(frame['q'],q) or not np.array_equal(frame['dq'],dq)):
                raise EpisodeRejected('Start frame does not match same real physics tick')
            self.start_receipt['first_eligible']=dict(frame_index=frame['index'],physics_row=row,
                time_s=float(state['time_s']),measured_q_rad=q.tolist(),measured_dq_rad_s=dq.tolist(),
                retained_command_q_rad=list(self.env.rows[-1]['q_command']))

    def _block(self,commands,subphase,*args,**kwargs):
        recovery=kwargs.get('recovery',args[1] if len(args)>1 else False)
        if not recovery and self.start_role in {'start_observation','eligible_start_route'}:
            values=np.asarray(commands,float)
            if values.ndim!=2 or values.shape[1:]!=(7,) or not np.isfinite(values).all():
                raise EpisodeRejected('Invalid start-route commands')
            if np.any(np.max(abs(values),axis=1)<=.03):
                raise EpisodeRejected('Eligible command would wash out declared start through home')
        return super()._block(commands,subphase,*args,**kwargs)

    def _move_to_prepose(self,row):
        if self.sequence_index!=0:return super()._move_to_prepose(row)
        self._point_plan=row
        if self.target.name!=self.start_manifest['target']:
            raise EpisodeRejected('Start manifest target differs from actual point')
        canonical,index=self._canonical(row)
        # The original immutable zero-home route is retained solely for the
        # original final return. It is never executed as the initial transfer.
        self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        if np.max(abs(self.env.state()['q']))>.03 or np.max(abs(self.q))>.03:
            raise EpisodeRejected('Preparation must begin at original dynamically reached home')
        requested=np.asarray(self.start_manifest['requested_start_q_rad'],float)
        deadline=time.monotonic()+30.
        for a,b in zip(canonical,canonical[1:]):
            if checked_edge(a,b,self.checker,deadline).get('clear') is not True:
                raise EpisodeRejected('Original canonical return route failed runtime preflight')
        if checked_edge(self.q,requested,self.checker,deadline).get('clear') is not True:
            raise EpisodeRejected('Start preparation edge failed runtime preflight')
        begin=len(self.env.rows);self.start_role='move_to_declared_start'
        try:self._joint_segment(requested,'initial_home')
        finally:self.start_receipt['preparation_physics_range']=[begin,len(self.env.rows)]
        begin=len(self.env.rows);self.start_role='settle_at_declared_start'
        try:self._hold(1.,'settle')
        finally:self.start_receipt['settle_physics_range']=[begin,len(self.env.rows)]
        state=self.env.state();checked_first_state(state,requested)
        total=sum(self.env.last_contacts.get(name,0.) for name in FRONT_GROUPS+FORBIDDEN_GROUPS)
        if (not np.isfinite([state['hall_N'],state['compression_m'],total]).all()
                or state['hall_N']>=.2 or total>=.2 or state['compression_m']>.00015
                or not self._observed_released(state)):
            raise EpisodeRejected('Declared start is not actually released')
        # Exactly twelve real physics ticks; the inherited camera hook naturally
        # emits its sole cadence frame. No extra capture or retroactive relabel.
        self.start_role='start_observation';self._hold(12/240.,'transfer')
        if self.start_receipt['first_eligible'] is None:
            raise EpisodeRejected('No natural-cadence first eligible frame')
        state=self.env.state();observed,_=checked_first_state(state,requested)
        record=dict(measured_planning_q_rad=observed.tolist(),retained_command_q_rad=self.q.tolist(),
            planning_physics_row=len(self.env.rows),planning_simulation_time_s=float(state['time_s']),
            planning_accepted=False,motion_complete=False,edge_checks=[])
        self.start_receipt['actual_start_route']=record
        started=time.monotonic();deadline=started+30.
        try:
            bridge,search=self._plan_inter_point_bridge(observed,canonical[index],deadline)
            record['search']=search
            if bridge is None:raise EpisodeRejected('Actual measured start bridge unavailable')
            route,stage=join_actual_route(self.q,observed,bridge,canonical,index)
            for a,b in zip(route,route[1:]):
                check=checked_edge(a,b,self.checker,deadline);record['edge_checks'].append(check)
                if check.get('clear') is not True:raise EpisodeRejected('Actual start route failed runtime preflight')
            if (len(self.env.rows)!=record['planning_physics_row']
                    or self.env.state()['time_s']!=record['planning_simulation_time_s']):
                raise EpisodeRejected('Physics advanced during start-route planning')
            record.update(joint_waypoints_q=route.tolist(),lower_staging_waypoint_index=stage,
                planning_accepted=True,original_contact_suffix_unchanged=True)
        finally:record['planning_wall_time_s']=time.monotonic()-started
        self.start_role='eligible_start_route'
        try:
            self._execute_staged(route,stage,initial=True)
            self._hold(1.,'align');record['motion_complete']=True
        finally:
            record['end_physics_row_exclusive']=len(self.env.rows);self.start_role=None
        return route

    def run(self,points):
        points=list(points)
        if len(points)!=1:raise ValueError('One independently reset target per bound start acquisition required')
        result=super().run(points)
        result.update(executed_controller_class=CLASS,candidate_method=CLASS,controller_source=SOURCE,
            episode_id=self.episode_id,start_family=self.start_manifest['start_family'],
            experiment_protocol_sha256=self.experiment_protocol_sha256,
            start_manifest_sha256=self.start_manifest_sha256,
            initial_route='real ineligible preparation then measured nonzero-start checked route; original contact and final home unchanged',
            qualification_passed=False,dataset_sealed=False)
        return result

    def finalize_start_receipt(self,sensor_frames):
        if self.start_receipt['first_eligible'] is None:raise ValueError('No actual first eligible frame to certify')
        result=dict(self.start_receipt)
        result.update(sensor_frames_sha256=hashlib.sha256(Path(sensor_frames).read_bytes()).hexdigest(),
                      qualification_passed=False,dataset_sealed=False)
        return result
