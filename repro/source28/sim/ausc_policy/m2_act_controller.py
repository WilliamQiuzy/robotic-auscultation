"""M2 state17 ACT L2 adapter; current FreshExpert owns every physical gate.

Integration (not automatically enabled): replace run_probe's FreshExpert(...)
construction by make_controller(..., deployment=config['m2_act_deployment'],
estimate_provider=provider). Keep the existing constructor kwargs. The provider
is called as provider(controller, state) and returns the exact fields checked
by validate_estimate; it must acquire an estimated target, never fill from the
analytic target/plan. Register this source in the run snapshot and apply
tag_learned_result AFTER run_probe's final result.update (which currently writes
learned_policy=False). ExpertCapture is rejected: it has expert source IDs and
cannot label learned commands. A separately labeled capture integration is
still required. Neither this module nor its CPU tests qualify any rollout.

This adapter imports no Isaac/torch at import time. CPU checkpoint loading is
explicit in make_controller; no fixture checkpoint or automatic fallback.
Only the generic persistent JointTrajectory mechanics are reused; no retired
ACT state/action contract, normalization, phase encoder or policy is imported.
"""
from copy import deepcopy
import math
import re
import time
import numpy as np

from . import m2_act as act, m2_contract as contract, m2_stage as stage
from .contact_entry import check_retained_pose
from .fresh_expert import FreshExpert
from sim.auscultation_expert_layout_v1.controller import DT, CONTROL_STEPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from sim.auscultation_pi05_v6.execution import JointTrajectory

DEPLOYMENT_SCHEMA='m2_act_l2_history2_deployment_v2'
CLASS=__name__+'.M2ACTController'
LIMITS=dict(velocity=.175,acceleration=.5,jerk=1.5)
L2_TIMEOUT_S=120.
TIME_TOLERANCE_S=1e-8
LABELS=dict(learned_policy=True,expert_demonstration=False,training_accepted=False,
            training_eligible=False,training_dataset_eligible=False)


def need(ok,message):
    if not ok:raise ValueError(message)


def validate_estimate(record,state,*,episode_id,target_id,metadata):
    """A same-time acquired estimate, with identity/revision and explicit phase.

    Phase is supplied by the deployment estimator/supervisor, not inferred from
    a teacher route index. It is not a numerical model feature. No Hall, dq,
    contact force, depth or RGB appears in the model state17.
    """
    fields={'episode_id','target_id','revision','phase','observed_at_s','measured_q',
            'estimated_target_base','confidence','source_id'}
    need(isinstance(record,dict) and set(record)==fields,'Explicit estimate envelope required')
    need(record['episode_id']==episode_id and record['target_id']==target_id,'Estimate identity mismatch')
    need(type(record['revision']) is int and record['revision']>=0,'Invalid estimate revision')
    need(record['phase'] in contract.L2_PHASES,'Model may only own FREE/APPROACH')
    stamp=contract._scalar(record['observed_at_s'],'estimate timestamp')
    now=contract._scalar(state['time_s'],'physics timestamp')
    need(abs(stamp-now)<=TIME_TOLERANCE_S,'Estimate must be acquired at current physics time')
    q=contract._array(record['measured_q'],(7,),'estimate measured_q')
    actual=contract._array(state['q'],(7,),'actual measured_q')
    need(np.array_equal(q,actual),'Estimate joints differ from same-time feedback')
    need(isinstance(record['source_id'],str) and 0<len(record['source_id'])<=256,'Missing estimate source ID')
    state17=contract.encode_state(q,record['estimated_target_base'],record['confidence'],metadata=metadata)
    need(state17[-1]>=.8,'Target confidence below deployment safety threshold')
    return dict(record,state17=state17,context=(episode_id,target_id,record['revision'],record['phase']))


class FixedOriginPrefix:
    """Decode H20 once; only E10 may dispatch on the original 20 Hz clock."""
    def __init__(self,block,*,observed_q,context,stamp):
        need(isinstance(block,contract.JointActionBlock) and block.arm=='block_relative_joint',
             'M2 deployment requires block-relative H20')
        need(np.array_equal(block.observed_q,observed_q),'Model returned a different block origin')
        self.absolute=contract.decode_joint_chunk(block)
        self.origin=block.observed_q.copy();self.context=context;self.stamp=float(stamp);self.index=0

    @property
    def remaining(self):return max(0,contract.EXECUTION_PREFIX-self.index)

    def clear(self):self.index=contract.EXECUTION_PREFIX

    def pop(self,stamp,context):
        need(self.remaining>0,'M2 prefix exhausted')
        if context!=self.context or not np.isfinite(stamp) or abs(stamp-self.stamp-self.index*.05)>TIME_TOLERANCE_S:
            self.clear();raise ValueError('M2 prefix context/clock changed')
        result=self.absolute[self.index].copy();self.index+=1;return result


def approach_gate(metrics,config):
    """Exact existing PhaseManager APPROACH geometry/speed predicate, read-only."""
    keys=('gap_m','tangent_m','angle_deg','tcp_speed_m_s')
    values=np.asarray([metrics[k] for k in keys],float)
    need(np.isfinite(values).all(),'Nonfinite handoff metrics')
    error=abs(float(metrics['gap_m'])-config.precontact_gap_m)
    passed=((error<=config.precontact_tolerance_m or
             math.isclose(error,config.precontact_tolerance_m,rel_tol=1e-12)) and
            0<=metrics['tangent_m']<=config.approach_tangential_tolerance_m and
            0<=metrics['angle_deg']<=config.angle_tolerance_deg and
            0<=metrics['tcp_speed_m_s']<config.approach_speed_limit_m_s)
    return dict(metrics,passed=bool(passed),normal_error_mm=1000*error,
                tangent_error_mm=1000*metrics['tangent_m'],angle_error_deg=metrics['angle_deg'])


def tag_learned_result(result,policy_receipt,handoffs):
    """Apply after outer run_probe metadata; physical success remains independent."""
    for row in [result,*result.get('points',[])]:row.update(LABELS)
    attempts=len(handoffs);successes=sum(x.get('l2_handoff_success') is True for x in handoffs)
    result.update(controller_source='M2_state17_ACT_L2 + unchanged_analytic_L3',
        executed_controller_class=CLASS,learned_policy_receipt=deepcopy(policy_receipt),
        l2_handoffs=deepcopy(handoffs),l2_handoff_attempt_count=attempts,l2_handoff_success_count=successes,
        l2_handoff_success_rate=successes/attempts if attempts else None,
        action='state17 causal history2; block-relative H20/E10 at20Hz; persistent bounded240Hz interpolation',
        observation_contract=contract.observation_contract(),
        scope='Learned L2 rollout; physical audit remains mandatory; never expert training data')
    configured=policy_receipt.get('configured_deployment')
    if (policy_receipt.get('schema')=='m2_act_pilot_history2_deployment_v2' or
            isinstance(configured,dict) and configured.get('schema')=='m2_act_pilot_history2_deployment_v2'):
        from .m2_pilot_deployment import FLAGS
        for row in [result,*result.get('points',[])]:row.update(FLAGS)
        result.update(diagnostic_closed_loop_evaluated=result.get('physics_steps',0)>0,
            scope='Pilot diagnostic on reused VALIDATION identities; no formal population/policy qualification; never expert data')
    return result


def make_controller(*args,deployment,estimate_provider,**kwargs):
    """Explicit production CPU factory; no data reads or fixture escape hatch."""
    fields={'schema','checkpoint','checkpoint_sha256','lerobot_root','metadata','episode_id'}
    need(isinstance(deployment,dict) and set(deployment)==fields and
         deployment['schema']==DEPLOYMENT_SCHEMA,'Invalid explicit M2 deployment configuration')
    need(isinstance(deployment['checkpoint_sha256'],str) and
         re.fullmatch('[0-9a-f]{64}',deployment['checkpoint_sha256']) is not None,'Invalid checkpoint SHA256')
    contract.require_observation_contract(deployment['metadata'])
    need(deployment['metadata'].get('target_source')!='given_contact_pose',
         'Given-contact task requires its separate deployment adapter')
    need(isinstance(deployment['episode_id'],str) and 0<len(deployment['episode_id'])<=128,'Episode identity required')
    predictor=act.load_checkpoint(deployment['checkpoint'],deployment['checkpoint_sha256'],
        lerobot_root=deployment['lerobot_root'],device='cpu',allow_cpu_fixture=False)
    receipt=dict(schema=DEPLOYMENT_SCHEMA,checkpoint_sha256=deployment['checkpoint_sha256'],
        metadata=deepcopy(deployment['metadata']),observation_contract=contract.observation_contract(),
        state_only=True,horizon=20,execution_prefix=10,
        control_hz=20,physics_hz=240,device='cpu',learned_policy=True,expert_demonstration=False,
        checkpoint_qualification_is_not_rollout_qualification=True)
    return M2ACTController(*args,predictor=predictor,metadata=deployment['metadata'],
        episode_id=deployment['episode_id'],estimate_provider=estimate_provider,policy_receipt=receipt,**kwargs)


class M2ACTController(FreshExpert):
    def __init__(self,*args,predictor,metadata,episode_id,estimate_provider,policy_receipt,
                 trajectory_factory=JointTrajectory,**kwargs):
        need(callable(predictor) and callable(estimate_provider),'Explicit predictor and acquired estimate provider required')
        contract.require_observation_contract(metadata)
        if metadata.get('target_source')=='given_contact_pose':
            need(getattr(self,'task_id',None)=='m2_given_contact_pose_control_v1',
                 'Given-contact semantics require the new controller adapter')
        need(kwargs.get('action_provider') is None,'L3 must retain its analytic actor')
        self.predictor=predictor;self.metadata=deepcopy(metadata);self.episode_id=episode_id
        self.estimate_provider=estimate_provider;self.policy_receipt=deepcopy(policy_receipt)
        self.trajectory_factory=trajectory_factory;self.trajectory=None;self.prefix=None
        self.l2_queries=[];self.l2_actions=[];self.l2_handoffs=[]
        self.control_owner='PREPARATION';self.actor_phase=None;self._l2_started_at=None
        self._history=None
        super().__init__(*args,**kwargs)
        need(getattr(self.env,'raw_capture',None) is None,
             'Expert capture cannot label learned commands; separate learned capture integration required')
        need(self.joint_speed_limit==.175,'M2 requires unchanged .175 rad/s limit')

    def _observation(self):
        state=self.env.state();count=len(self.env.rows);stamp=float(state['time_s'])
        record=self.estimate_provider(self,state)
        need(len(self.env.rows)==count and abs(float(self.env.state()['time_s'])-stamp)<=TIME_TOLERANCE_S,
             'Physics advanced during target acquisition')
        record=validate_estimate(record,self.env.state(),episode_id=self.episode_id,
            target_id=self.target.name,metadata=self.metadata)
        if self.actor_phase=='APPROACH' and record['phase']=='FREE':
            raise EpisodeRejected('APPROACH cannot regress to FREE within a target')
        self.actor_phase=record['phase']
        if self.control_owner=='ACT_L2':
            if self._history is None:self._history=stage.CausalHistory(self._l2_started_at)
            self._history.observe(record['state17'],stamp,record['context'])
        else:
            self._history=None
        need(self.manager.state.phase==Phase.APPROACH and self.manager.state.failure_reason is None,
             'L2 no longer owns the analytic precontact phase')
        return record

    def _deadline(self):
        stamp=contract._scalar(self.env.state()['time_s'],'physics time')
        need(self._l2_started_at is not None and 0<=stamp-self._l2_started_at<L2_TIMEOUT_S,
             'M2 L2 clock/120 second deadline failed')

    def _query(self,observation):
        need(self.control_owner=='ACT_L2','Model cannot run after L3 handoff')
        self._deadline();state=self.env.state();stamp=float(state['time_s']);count=len(self.env.rows)
        observed_q=np.asarray(state['q']).copy();self.prefix=None
        need(self._history is not None,'Two real same-context history observations required; no padding')
        history,history_times=self._history.snapshot()
        need(abs(history_times[-1]-stamp)<=TIME_TOLERANCE_S and
             np.array_equal(history[-1],observation['state17']),'Inference history must end at current observation')
        row=dict(index=len(self.l2_queries),time_s=stamp,physics_row=count,
            context=list(observation['context']),source_id=observation['source_id'],
            state17=observation['state17'].tolist(),history=history.tolist(),history_times=history_times.tolist(),
            observation_contract=contract.observation_contract(self.metadata),accepted=False,**LABELS)
        self.l2_queries.append(row);started=time.monotonic()
        try:
            block=self.predictor(observed_q.copy(),observation['estimated_target_base'],
                observation['confidence'],metadata=self.metadata,history=history,history_times=history_times)
            need(isinstance(block,contract.JointActionBlock) and all(
                block.metadata.get(key)==self.metadata.get(key) for key in
                ('contract_id','target_source','observation_contract','reference_sha256','reference_revision')),
                'Model action block semantic route/reference mismatch')
            need(len(self.env.rows)==count and abs(self.env.state()['time_s']-stamp)<=TIME_TOLERANCE_S and
                 np.array_equal(self.env.state()['q'],observed_q),'Physics/feedback changed during inference')
            self.prefix=FixedOriginPrefix(block,observed_q=observation['state17'][:7],
                context=observation['context'],stamp=stamp)
            need(np.all(self.prefix.absolute>=self.kin.lower_limits) and
                 np.all(self.prefix.absolute<=self.kin.upper_limits),'Raw ACT chunk outside joint limits')
            row.update(accepted=True,block_origin_q=self.prefix.origin.tolist(),absolute_h20=self.prefix.absolute.tolist())
        except Exception as error:
            self.prefix=None;row['failure_reason']=str(error);raise
        finally:row['inference_wall_s']=time.monotonic()-started

    def _advance_tick(self,subphase):
        self._deadline();stamp=float(self.env.state()['time_s']);count=len(self.env.rows)
        command=self.trajectory.step()
        # Original FreshExpert -> SkillExpert -> posture/static-world/physical
        # checks run for every command AND observed state. Never call env.step.
        self._block(np.asarray(command)[None],subphase,active=False)
        need(len(self.env.rows)==count+1 and abs(self.env.state()['time_s']-stamp-DT)<=TIME_TOLERANCE_S,
             'Nonconsecutive native physics clock/rows')
        need(np.array_equal(self.env.rows[-1]['q_command'],command),'Native command differs from interpolator')

    def _execute_action(self,context):
        self._deadline();stamp=float(self.env.state()['time_s']);index=self.prefix.index
        proposal=self.prefix.pop(stamp,context)
        row=dict(time_s=stamp,target=self.target.name,phase=context[-1],prefix_index=index,
            query_index=len(self.l2_queries)-1,proposal=proposal.tolist(),completed=False,
            physics_row_start=len(self.env.rows),**LABELS)
        self.l2_actions.append(row);self.trajectory.set_target(proposal)
        try:
            for _ in range(CONTROL_STEPS):
                self._advance_tick('m2_act_l2_'+context[-1].lower())
                observation=self._observation()
                if observation['context']!=context:
                    self.prefix=None;row['interrupted']='context';return 'context'
                if self._handoff_measurements()['passed']:
                    self.prefix=None;row['interrupted']='handoff';return 'handoff'
            need(abs(self.env.state()['time_s']-stamp-contract.PERIOD_S)<=TIME_TOLERANCE_S,'20Hz action clock mismatch')
            row['completed']=True;return 'continue'
        except Exception as error:
            self.prefix=None;row['failure_reason']=str(error);raise
        finally:row['physics_row_end_exclusive']=len(self.env.rows)

    def _handoff_measurements(self):
        state=self.env.state();metric=self.metrics(state)
        speed=self.env.rows[-1].get('tcp_speed_m_s') if self.env.rows else None
        if speed is None:return dict(passed=False,reason='No native measured TCP speed')
        gate=approach_gate(dict(metric,tcp_speed_m_s=speed),self.manager.config)
        retained=check_retained_pose(self.kin.tcp(self.q),self.target.position_base,
            self.normal_base,self.rotation_base,position_tolerance_m=self.executor_config['curve_position_tolerance_m'],
            angle_tolerance_rad=self.executor_config['curve_angle_tolerance_rad'])
        # Full curve construction will be repeated by unchanged L3; this cannot
        # substitute an alignment motion, IK correction or alternate target.
        return dict(passed=bool(self.actor_phase=='APPROACH' and gate['passed'] and retained['passed']),
            measured=gate,retained_curve_entry=retained,time_s=float(state['time_s']),
            physics_row=len(self.env.rows),command_q=self.q.tolist(),measured_q=np.asarray(state['q']).tolist())

    def _stationary(self):
        if len(self.env.rows)<3:return False
        values=np.asarray([row['q_command'] for row in self.env.rows[-3:]],float)
        return bool(values.shape==(3,7) and np.array_equal(values,np.tile(self.q,(3,1))))

    def _stop_for_handoff(self,context):
        self.prefix=None;self.control_owner='L2_STOP';self.trajectory.set_target(self.q.copy())
        for _ in range(5*240):
            self._advance_tick('m2_l2_handoff_stop')
            need(self._observation()['context']==context,'Estimate context changed during handoff stop')
            if (np.max(abs(self.trajectory.velocity))<=1e-10 and
                    np.max(abs(self.trajectory.acceleration))<=1e-10 and self._stationary()):
                entry=self._handoff_measurements()
                need(entry['passed'],'Analytic entry lost during bounded stop')
                self.control_owner='L3_ENTRY';return entry
        raise EpisodeRejected('M2 bounded handoff stop timeout')

    def _move_to_prepose(self,row):
        self._point_plan=row
        canonical,index=self._canonical(row)
        self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        receipt=dict(target=self.target.name,l2_handoff_success=False,l3_started=False,l3_completed=False,
                     expert_l2_fallback=False,**LABELS)
        self.l2_handoffs.append(receipt)
        try:
            # Bounded single-point deployment matches the approved heldout
            # single-point evaluation denominator. No analytic transfer command.
            need(self.sequence_index==0 and np.max(abs(self.q))<=.03 and
                 np.max(abs(self.env.state()['q']))<=.03,'M2 episode must start at original home')
            self.control_owner='PREPARATION';self._hold(1.,'m2_initial_stationary_settle')
            need(self._stationary(),'No real stationary history for interpolator initialization')
            self.trajectory=self.trajectory_factory(self.q,dt=DT,**LIMITS)
            self.actor_phase='FREE';self.control_owner='ACT_L2';self.prefix=None
            self._l2_started_at=float(self.env.state()['time_s'])
            self._history=stage.CausalHistory(self._l2_started_at)
            receipt.update(start_time_s=self._l2_started_at,start_physics_row=len(self.env.rows))
            while True:
                self._deadline();observation=self._observation();context=observation['context']
                if self._handoff_measurements()['passed']:
                    entry=self._stop_for_handoff(context)
                    receipt.update(l2_handoff_success=True,entry=entry,context=list(context),
                        query_count_at_handoff=len(self.l2_queries));return canonical
                if self.prefix is None or not self.prefix.remaining or self.prefix.context!=context:
                    self.prefix=None
                    if self._history is None or len(self._history.frames)!=2:
                        self.trajectory.set_target(self.q.copy())
                        self._advance_tick('m2_l2_history_wait')
                        continue
                    self._query(observation)
                self._execute_action(context)
        except Exception as error:
            receipt['failure_reason']=str(error);self.prefix=None;self.control_owner='STOPPED';raise
        finally:
            receipt['end_physics_row']=len(self.env.rows)
            try:receipt['last_measurement']=self._handoff_measurements()
            except Exception as error:receipt['last_measurement_error']=str(error)

    def _contact_cycle(self):
        need(self.l2_handoffs and self.l2_handoffs[-1]['l2_handoff_success'],'No verified L2 handoff')
        receipt=self.l2_handoffs[-1];entry=receipt['entry']
        need(self.control_owner=='L3_ENTRY' and self.prefix is None and self._stationary() and
             len(self.env.rows)==entry['physics_row'] and
             abs(self.env.state()['time_s']-entry['time_s'])<=TIME_TOLERANCE_S and
             np.array_equal(self.q,entry['command_q']) and
             len(self.l2_queries)==receipt['query_count_at_handoff'] and self._handoff_measurements()['passed'],
             'Handoff changed before analytic L3')
        self.control_owner='ANALYTIC_L3';receipt['l3_started']=True
        try:
            super()._contact_cycle()
            receipt['l3_completed']=True
        except Exception as error:
            receipt['l3_failure_reason']=str(error);raise
        finally:
            self.prefix=None;receipt['l3_end_physics_row']=len(self.env.rows)

    def _tick(self,*args,**kwargs):
        start=len(self.env.rows)
        try:return super()._tick(*args,**kwargs)
        finally:
            for row in self.env.rows[start:]:row.update(LABELS,control_owner=self.control_owner,
                command_source_id='m2_act_l2_interpolated' if self.control_owner in ('ACT_L2','L2_STOP') else 'analytic_l3_or_preparation')
            if self.control_rows:self.control_rows[-1].update(LABELS,control_owner=self.control_owner)

    def run(self,points):
        points=list(points);need(len(points)==1,'M2 deployment requires one declared target per episode')
        try:
            result=super().run(points)
            result.update(l2_queries=self.l2_queries,l2_actions=self.l2_actions,
                interpolation='persistent bounded JointTrajectory; proposals are not promised achieved endpoints',
                learned_joint_command_limits=LIMITS.copy(),l2_timeout_s=L2_TIMEOUT_S)
            for key in ('candidate_policy','candidate_geometry','candidate_timing','route_selection','candidate_scope'):
                result.pop(key,None)
            return tag_learned_result(result,self.policy_receipt,self.l2_handoffs)
        finally:
            for row in [*self.env.rows,*self.control_rows]:row.update(LABELS)
