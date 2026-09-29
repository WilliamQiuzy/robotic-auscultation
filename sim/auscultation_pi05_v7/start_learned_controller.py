"""Versioned nonzero-start preparation before the unchanged absolute-q policy.

Only zero-to-declared-start is classical preparation. No expert start-to-staging
route is consulted. Contact, safety, inter-point motion and final home retain
the frozen V6 implementation. This class is not the original V6 controller.
"""
import hashlib
from pathlib import Path
import time
import numpy as np

from sim.auscultation_pi05_v6.learned_controller import LearnedController
from sim.auscultation_pi05_v6.execution import JointTrajectory
from sim.auscultation_pi05_v6 import phase_contract
from sim.auscultation_pi05_v6.supervision import contact_load_estimate
from sim.auscultation_expert_layout_v1.controller import EpisodeRejected, FRONT_GROUPS, FORBIDDEN_GROUPS
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from sim.expert_quality_20260924.candidate_planner import checked_edge

CLASS='sim.auscultation_pi05_v7.start_learned_controller.StartLearnedController'
SOURCE='pi05_absolute_joint_policy_with_bound_start_preparation'
RECEIPT_SCHEMA='wam_v7_learned_measured_start_receipt_v1'
START_POSITION_TOLERANCE_RAD=.03
START_SPEED_LIMIT_RAD_S=.03


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def checked_start_state(state,requested):
    q=np.asarray(state['q'],float);dq=np.asarray(state['dq'],float);requested=np.asarray(requested,float)
    if (q.shape!=(7,) or dq.shape!=(7,) or requested.shape!=(7,)
            or not np.isfinite(np.r_[q,dq,requested,state['time_s']]).all()):
        raise ValueError('Nonfinite or malformed measured start')
    if np.max(abs(q-requested))>START_POSITION_TOLERANCE_RAD or np.max(abs(q))<=.03:
        raise EpisodeRejected('Measured start misses declared request or washed out to home')
    if np.max(abs(dq))>START_SPEED_LIMIT_RAD_S:
        raise EpisodeRejected('Measured start has not settled below the declared speed gate')
    return q,dq


def new_receipt(start,manifest_sha,protocol_sha,episode_id):
    return dict(schema=RECEIPT_SCHEMA,episode_id=episode_id,target=start['target'],
        start_family=start['start_family'],start_manifest_sha256=manifest_sha,
        experiment_protocol_sha256=protocol_sha,source_plan_sha256=start['parent_plan_sha256'],
        requested_start_q_rad=start['requested_start_q_rad'],executed_controller_class=CLASS,
        controller_source=SOURCE,controller_source_sha256=sha(__file__),
        start_position_tolerance_rad=START_POSITION_TOLERANCE_RAD,
        start_speed_limit_rad_s=START_SPEED_LIMIT_RAD_S,
        preparation_origin=None,preparation_physics_range=None,settle_physics_range=None,observation_physics_range=None,
        learned_interval_start_physics_row=None,first_query=None,
        preparation_excluded_from_learned_interval=True,qualification_passed=False,
        scope='Producer provenance only; independent start, action, supervisor and physical audits required')


class StartLearnedController(LearnedController):
    def __init__(self,*args,start_manifest,start_manifest_sha256,experiment_protocol_sha256,
                 episode_id,**kwargs):
        if kwargs.get('task_scope','full')!='full':
            raise ValueError('Bound start evaluation requires all learned phases')
        self.start_manifest=start_manifest;self.start_manifest_sha256=start_manifest_sha256
        self.experiment_protocol_sha256=experiment_protocol_sha256;self.episode_id=episode_id
        self.start_role=None;self.start_prepared=False
        self.start_receipt=new_receipt(start_manifest,start_manifest_sha256,experiment_protocol_sha256,episode_id)
        super().__init__(*args,**kwargs)

    def _capture_observed_tick(self,state,subphase,active,recovery):
        if self.start_role is not None and not recovery:
            if active or subphase not in {'initial_home','settle'}:
                raise EpisodeRejected('Classical preparation escaped its declared inactive interval')
            self.env.rows[-1]['preparation_role']=self.start_role
        return super()._capture_observed_tick(state,subphase,active,recovery)

    def _require_released_start(self,state):
        checked_start_state(state,self.start_manifest['requested_start_q_rad'])
        _,hall,_=self._tracked_metrics(state)
        total=sum(self.env.last_contacts.get(name,0.) for name in FRONT_GROUPS+FORBIDDEN_GROUPS)
        if (not np.isfinite([state['hall_N'],state['compression_m'],total]).all()
                or state['hall_N']>=.2 or total>=.2 or state['compression_m']>.00015
                or contact_load_estimate(hall)>=.2 or hall.displacement_estimate_m>.00015):
            raise EpisodeRejected('Declared learned start is not actually released')

    def _bind_first_query(self,phase,active):
        if self.start_receipt['first_query'] is not None:return
        if (not self.start_prepared or self.sequence_index!=0 or self.start_role is not None
                or phase!='FREE' or active or self.target.name!=self.start_manifest['target']):
            raise EpisodeRejected('First learned query must be inactive FREE at the declared measured start')
        state=self.env.state();q,dq=checked_start_state(state,self.start_manifest['requested_start_q_rad'])
        self._require_released_start(state)
        row=len(self.env.rows)
        if row!=self.start_receipt['learned_interval_start_physics_row'] or not self.sensor.rows:
            raise EpisodeRejected('First query does not immediately follow bounded preparation')
        frame=self.sensor.rows[-1]
        if (frame['target']!=self.target.name or frame['physics_row']!=row-1
                or abs(frame['time_s']-state['time_s'])>1e-9
                or not np.array_equal(frame['q'],q) or not np.array_equal(frame['dq'],dq)):
            raise EpisodeRejected('First query raw frame does not match the actual measured start tick')
        raw=(self.sensor.out/frame['raw_file']).resolve()
        if (not raw.is_relative_to(self.sensor.out.resolve()) or not raw.is_file()
                or sha(raw)!=frame['raw_sha256']):
            raise EpisodeRejected('First query raw sensor bytes differ')
        self.start_receipt['first_query']=dict(role='first_learned_policy_decision',
            requested_policy_phase=phase,acquired_observation_phase=frame['phase'],
            acquired_frame_eligible=frame['eligible'],observation_frame=frame['index'],
            observation_physics_row=frame['physics_row'],physics_row_start=row,
            time_s=float(state['time_s']),measured_q_rad=q.tolist(),measured_dq_rad_s=dq.tolist(),
            retained_command_q_rad=self.q.tolist(),raw_sensor_file=frame['raw_file'],
            raw_sensor_sha256=frame['raw_sha256'],model_request_receipt_present=False)

    def _infer_step(self,phase,*,active):
        self._bind_first_query(phase,active)
        # Original implementation composes raw[0] as an ABSOLUTE target. This
        # wrapper neither adds q nor supplies any alternate action or path.
        try:return super()._infer_step(phase,active=active)
        finally:
            first=self.start_receipt['first_query']
            if first is not None and self.learned_rows:
                record=self.learned_rows[0]
                first['learned_query_index']=record['index']
                first['model_request_receipt_present']=isinstance(record.get('inference'),dict)
                if first['model_request_receipt_present']:
                    first['request_id']=record['inference']['request_id']

    def _move_to_prepose(self,row):
        if self.sequence_index!=0:return super()._move_to_prepose(row)
        if self.start_prepared or self.task_scope!='full':
            raise EpisodeRejected('Start preparation may run exactly once for a full learned task')
        self._point_plan=row
        if self.target.name!=self.start_manifest['target']:
            raise EpisodeRejected('Start manifest target differs from the first actual point')
        canonical,index=self._canonical(row)
        self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        if np.max(abs(self.env.state()['q']))>.03 or np.max(abs(self.q))>.03:
            raise EpisodeRejected('Preparation must begin at original dynamically reached home')
        origin=self.env.state()
        if not np.isfinite(np.r_[origin['q'],origin['dq'],origin['time_s'],self.q]).all():
            raise EpisodeRejected('Nonfinite preparation origin')
        self.start_receipt['preparation_origin']=dict(physics_row_start=len(self.env.rows),
            time_s=float(origin['time_s']),measured_q_rad=np.asarray(origin['q']).tolist(),
            measured_dq_rad_s=np.asarray(origin['dq']).tolist(),retained_command_q_rad=self.q.tolist())
        requested=np.asarray(self.start_manifest['requested_start_q_rad'],float)
        deadline=time.monotonic()+30.
        for a,b in zip(canonical,canonical[1:]):
            if checked_edge(a,b,self.checker,deadline).get('clear') is not True:
                raise EpisodeRejected('Original final return route failed runtime preflight')
        if checked_edge(self.q,requested,self.checker,deadline).get('clear') is not True:
            raise EpisodeRejected('Declared start preparation edge failed runtime preflight')
        try:
            for field,role,operation in (
                ('preparation_physics_range','move_to_declared_start',lambda:self._joint_segment(requested,'initial_home')),
                ('settle_physics_range','settle_at_declared_start',lambda:self._hold(1.,'settle')),
                ('observation_physics_range','start_observation',lambda:self._hold(12/240.,'settle'))):
                begin=len(self.env.rows);self.start_role=role
                try:operation()
                finally:self.start_receipt[field]=[begin,len(self.env.rows)]
            self._require_released_start(self.env.state())
            self.start_prepared=True
            self.start_receipt['learned_interval_start_physics_row']=len(self.env.rows)
        finally:self.start_role=None
        # Same first-target V6 learned loop, with its zero-only preparation
        # replaced above. Canonical waypoints are retained for FINAL HOME only.
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
        self.actor_phase=None
        return canonical

    def run(self,points):
        points=list(points)
        if not points or points[0][0].name!=self.start_manifest['target']:
            raise ValueError('Declared start must bind the first requested target')
        result=super().run(points)
        result.update(executed_controller_class=CLASS,controller_source=SOURCE,
            episode_id=self.episode_id,start_family=self.start_manifest['start_family'],
            start_manifest_sha256=self.start_manifest_sha256,
            experiment_protocol_sha256=self.experiment_protocol_sha256,
            preparation_home='one classical zero-to-declared-start preparation; learned onward sequence; classical final home',
            dataset_role='learned_policy_evaluation_not_expert_data',expert_demonstration=False,
            training_dataset_eligible=False,training_accepted=False,
            start_acceptance_audit_required=True)
        return result

    def finalize_start_receipt(self):
        result=dict(self.start_receipt)
        result['complete']=bool(self.start_prepared and result['first_query'] is not None
                                and result['first_query']['model_request_receipt_present'])
        result['artifact_sha256']={}
        for name in ('expert_trace.jsonl','sensors/frames.jsonl','learned_actions.jsonl'):
            path=self.env.out/name
            if path.is_file():result['artifact_sha256'][name]=sha(path)
        return result
