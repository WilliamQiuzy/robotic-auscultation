"""Opt-in TRAIN DART collection on current 5mm/64-iteration native mechanics.

Prepare configs only (does not launch):
  python -m sim.ausc_policy.m2_given_contact_recovery --source-config FILE \
    --source-sha256 SHA --episode-id ID --seed 17 --amplitude .015 --output NEW.json

The owned geometry_qualification worker executes the resulting config later.
Perturbations are actual raw setters, excluded by the separate recovery exporter.
No old R4 oracle, teleport, interpolator reset, emergency recovery, or L2 fallback.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import time

import numpy as np

from . import m2_contract as contract, m2_given_contact_handoff as handoff
from . import m2_given_contact_runtime as runtime
from .m2_given_contact_controller import GivenContactExpert, GIVEN_TRAJECTORY_LIMITS
from .m2_given_contact_pose import GivenPoseProvider, need, require_route
from sim.auscultation_skill_v1.executor import command_derivatives
from sim.auscultation_pi05_v6.execution import FinishedStationaryRoundoffError

ROUTE_KEY='m2_given_contact_recovery'
SCHEMA='m2_given_contact_dart_collection_v1'
LATE_SCHEMA='m2_given_contact_late_dart_collection_v2'
STUDENT_SCHEMA='m2_given_contact_student_prefix_recovery_v3'
SIDECAR_SCHEMA='m2_given_contact_recovery_setter_roles_v1'
NATIVE_PYTHON='/opt/wam-runtime/envs/isaacsim60/bin/python'
PHYSICS=dict(position_iterations=64,velocity_iterations=1,armature_kg_m2=.02)
PERTURB_CELLS=10
APPROACH_DELAY_CELLS=4
MAX_PREVIEW_STEPS=120*240
# Offline data collection pauses physics while the unchanged full-mesh checker
# previews every command. This CPU budget is separate from physical/ALIGN limits.
MAX_PLAN_CPU_S=300.
KINDS=frozenset(('nominal_teacher','recovery_teacher','perturbation','excluded'))


def _settings(value):
    if isinstance(value,dict) and value.get('schema')==STUDENT_SCHEMA:
        need(set(value)=={'schema','source_config','episode_id','student_prefix'},
            'Exact explicit historical-student prefix settings required')
        need(isinstance(value['episode_id'],str) and
            re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',value['episode_id']) is not None,
            'Bounded distinct recovery episode required')
        runtime.bound_path(value['student_prefix'])
        return value
    late=isinstance(value,dict) and value.get('schema')==LATE_SCHEMA
    keys={'schema','source_config','episode_id','seed','amplitude_rad'}|({'trigger_distance_m'} if late else set())
    need(isinstance(value,dict) and set(value)==keys and value.get('schema') in (SCHEMA,LATE_SCHEMA),
         'Exact explicit DART settings required')
    if late:need(type(value['trigger_distance_m']) in (int,float) and value['trigger_distance_m'] in (.035,.060),
        'Predeclared 35/60mm late approach trigger required')
    need(type(value['seed']) is int and 0<=value['seed']<2**32 and
         type(value['amplitude_rad']) in (int,float) and value['amplitude_rad'] in (.015,.03),
         'Fixed seeded .015/.03 rad perturbation required')
    need(isinstance(value['episode_id'],str) and
         re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',value['episode_id']) is not None,
         'Bounded distinct recovery episode required')
    return value


def _project(original, settings):
    result=deepcopy(original);route=result['m2_given_contact_control'];episode=settings['episode_id']
    need(episode!=route['episode_id'], 'Recovery requires a fresh episode identity')
    route['episode_id']=episode;route['metadata']['episode_id']=episode
    capture=result['fresh_expert']['raw_multimodal_capture']
    capture['episode_id']=episode;capture['compact']['metadata']['episode_id']=episode
    if 'compact_scene_storage' in result:result['compact_scene_storage']['batch_id']=episode
    result[ROUTE_KEY]=deepcopy(settings)
    return result


def preflight(config):
    """Authenticate TRAIN intent and exact original scene projection, no USD."""
    meta=config.get('m2_given_contact_control',{}).get('metadata',{})
    need(meta.get('split')=='train', 'Recovery collection is TRAIN only; VAL/TEST stay sealed')
    runtime.shared.require_membership('train',meta.get('subject_id'))
    settings=_settings(config.get(ROUTE_KEY))
    need(not any(k in config for k in ('m2_act_deployment','m2_given_contact_deployment',
        'm2_given_contact_pilot_deployment','m2_given_contact_expanded_pilot_deployment','m2_given_contact_evaluation')),
        'Recovery teacher cannot mix learned/formal/TEST routes')
    need(runtime.digest(config.get('physics'))==runtime.digest(PHYSICS) and config.get('isaac_python')==NATIVE_PYTHON and
         config.get('geometry_id')==runtime.geometry.GEOMETRY_ID and
         config.get('geometry_source_sha256')==runtime.geometry.GEOMETRY_SOURCE_SHA256,
         'Current 64-iteration Native60/5mm geometry required')
    original=runtime.read_bound(settings['source_config'])
    need(ROUTE_KEY not in original and original.get('m2_given_contact_control',{}).get('metadata',{}).get('split')=='train',
         'Original nominal TRAIN teacher config required')
    need(runtime.digest(config)==runtime.digest(_project(original,settings)),
         'Recovery altered original scene/force/config projection')
    route=config['m2_given_contact_control'];ref=require_route(meta);ref.verify_sources()
    need(route.get('schema')==runtime.pose.TASK_ID and route.get('handoff_policy_id')==handoff.POLICY_ID and
         meta.get('handoff_policy_id')==handoff.POLICY_ID and route['episode_id']==settings['episode_id']==meta['episode_id'],
         'Original given reference/interior parking contract required')
    capture=config['fresh_expert']['raw_multimodal_capture']
    need(capture.get('enabled') is True and capture.get('profile')=='compact' and
         capture.get('episode_id')==settings['episode_id'] and capture['compact']['metadata']==meta and
         config['fresh_expert'].get('plans')==[ref.snapshot()['sources']['fresh_plan']],
         'Exact native compact/reference/plan binding required')
    result=dict(metadata=deepcopy(meta),settings=deepcopy(settings),reference=ref,config_sha256=runtime.digest(config))
    if settings['schema']==STUDENT_SCHEMA:
        from . import m2_student_recovery_prefix as student
        result['student_prefix']=student.load(settings['student_prefix'],settings['source_config'])
    return result


def derive_config(source_config, *, episode_id, seed, amplitude_rad, trigger_distance_m=None):
    settings=dict(schema=SCHEMA if trigger_distance_m is None else LATE_SCHEMA,source_config=deepcopy(source_config),episode_id=episode_id,
        seed=seed,amplitude_rad=amplitude_rad)
    if trigger_distance_m is not None:settings['trigger_distance_m']=trigger_distance_m
    _settings(settings);original=runtime.read_bound(source_config)
    result=_project(original,settings);preflight(result);return result


def derive_student_config(source_config,*,episode_id,student_prefix):
    settings=dict(schema=STUDENT_SCHEMA,source_config=deepcopy(source_config),episode_id=episode_id,
        student_prefix=deepcopy(student_prefix))
    _settings(settings);result=_project(runtime.read_bound(source_config),settings);preflight(result);return result


def perturbation_cell_count(settings):
    if settings.get('schema')!=STUDENT_SCHEMA:return PERTURB_CELLS
    from . import m2_student_recovery_prefix as student
    prefix=runtime.read_bound(settings['student_prefix']);targets=np.asarray(prefix.get('raw_targets'),float)
    need(prefix.get('schema')==student.SCHEMA and targets.ndim==2 and targets.shape[1:]==(7,) and
        1<=len(targets)<=2400 and np.isfinite(targets).all(),'Bounded authenticated student prefix cells required')
    return len(targets)


def recovery_trigger(settings, *, phase, approach_cells, distance_m):
    if settings['schema']==STUDENT_SCHEMA:return phase in ('FREE','APPROACH') and approach_cells>=1
    if phase!='APPROACH':return False
    if settings['schema']==SCHEMA:return approach_cells>=APPROACH_DELAY_CELLS
    need(settings['schema']==LATE_SCHEMA and np.isfinite(distance_m) and distance_m>=0,
         'Finite measured late-recovery distance required')
    return distance_m<=settings['trigger_distance_m']


def _preview(trajectory, history, targets, verify_command, *, settle, trajectory_factory=None, deadline=None):
    """Clone retained command q/v/a and inspect every proposed 240Hz sample."""
    deadline=time.monotonic()+MAX_PLAN_CPU_S if deadline is None else deadline
    original=handoff.input_snapshot(trajectory);original_output=handoff.output_snapshot(trajectory)
    history=contract._array(history,(3,7),'actual command history').copy()
    need(np.array_equal(history[-1],original['fields']['current_position']), 'Preview command seam differs')
    clone=handoff.restore_trajectory(original,trajectory_factory or type(trajectory))
    commands=[];cells=[];peaks={k:0. for k in GIVEN_TRAJECTORY_LIMITS}
    try:
        for target in targets:
            target=contract._array(target,(7,),'raw target proposal')
            while True:
                need(time.monotonic()<deadline and len(commands)+12<=MAX_PREVIEW_STEPS,
                     'Recovery preview deadline/120 second command bound')
                clone.set_target(target);cells.append(target.tolist())
                for _ in range(12):
                    need(time.monotonic()<deadline,'Recovery full-path verification deadline')
                    q=contract._array(clone.step(),(7,),'preview command')
                    values=command_derivatives(history,q)
                    need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and
                         values['jerk']<=1.5+1e-8, 'Recovery preview original derivative seam/limits failed')
                    for key in peaks:peaks[key]=max(peaks[key],values[key])
                    verify_command(q);commands.append(q.copy());history=np.vstack([history[1:],q])
                if not settle or (np.max(abs(q-target))<=1e-10 and
                        np.max(abs(clone.velocity))<=1e-10 and np.max(abs(clone.acceleration))<=1e-10):break
    finally:
        need(handoff.input_snapshot(trajectory)==original and handoff.output_snapshot(trajectory)==original_output,
             'Preview changed live trajectory input/output')
    commands=np.asarray(commands,dtype=np.float64).reshape(-1,7)
    return dict(input_state=original,input_sha256=runtime.digest(original),cells=cells,commands=commands.tolist(),
        commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),steps=len(commands),
        command_derivative_maxima=peaks,all_240hz_commands_verified=True,physics_executed=False)


def preview_cells(trajectory, command_history, targets, verify_command, **kwargs):
    return _preview(trajectory,command_history,targets,verify_command,settle=False,**kwargs)


def _validate_query(query, reference,*,replay_start=False):
    q=contract._array(query['measured_q'],(7,),'same-state measured q')
    dq=contract._array(query['measured_dq'],(7,),'same-state measured dq')
    history=contract._array(query['history'],(2,17),'real recovery history')
    times=contract._array(query['history_times'],(2,),'real recovery clocks')
    context=query['context'];stamp=query['time_s'];row=query['physics_row']
    need(isinstance(context,(tuple,list)) and len(context)==4,'Same-state context malformed')
    metadata=query['metadata'];need(metadata.get('split')=='train','Same-state recovery is TRAIN only')
    runtime.shared.require_membership('train',metadata.get('subject_id'))
    need(require_route(metadata).sha256==reference.sha256 and context[0]==metadata['episode_id'],
         'Same-state metadata/reference/episode differs')
    need(type(replay_start) is bool and query['reference_sha256']==reference.sha256 and len(context)==4 and type(context[2]) is int and
         context[2]==0 and (context[3]=='APPROACH' or replay_start and context[3]=='FREE') and
         context[1]==reference.snapshot()['target_id'], 'Same-state reference/APPROACH context differs')
    need(type(row) is int and row>=12 and row%12==0 and abs(stamp-row/240.)<=1e-8 and
         abs(times[1]-stamp)<=1e-8 and abs(times[1]-times[0]-.05)<=1e-8 and
         np.array_equal(history[-1,:7],q), 'Same-state history/clock differs')
    need(np.max(abs(dq))<=.175, 'Same-state measured velocity exceeds original limit')
    for frame in history:
        encoded=contract.encode_state(frame[:7],reference.pose_base,1.,metadata=metadata)
        need(np.allclose(encoded,frame,atol=1e-10,rtol=0),'Same-state physically inconsistent state17 history')
    scalars=np.asarray([query[k] for k in ('hall_N','physical_contact_N','compression_m')],float)
    # Preserve raw lower-stop solver residuals. The inherited capture/Hall
    # validator owns their existing lower-endpoint domain; do not zero them.
    need(np.isfinite(scalars).all() and 0<=scalars[0]<.2 and 0<=scalars[1]<.2 and scalars[2]<=.00015,
         'Recovery requires actual released noncontact state')
    return q


def recovery_route_candidates(retained, route,*,all_joints=False):
    """Measured state plans geometry; retained command q/v/a plans dynamics.

    Do not require an exact zero-velocity stop at the noisy sensor sample.
    Tiny deterministic J5 detours can avoid a native Ruckig terminal degeneracy;
    every issued sample must still pass the original complete preview.
    """
    retained=contract._array(retained,(7,),'retained command')
    route=np.asarray(route,dtype=float)
    candidates=[np.vstack([retained,route[1:]])]
    for joint in ([4,0,1,2,3,5,6] if all_joints else [4]):
        for delta in (.001,-.001,.003,-.003):
            intermediate=route[0].copy();intermediate[joint]+=delta
            candidates.append(np.vstack([retained,intermediate,route[1:]]))
    return candidates


def record_incomplete_recovery(result):
    reason='No complete actual DART perturbation/recovery pair'
    result.update(success=False,recovery_qualification_failure=reason)
    if not result.get('failure_reason'):result['failure_reason']=reason


def propose_recovery(query, trajectory, command_history, destination, *, reference, planner,
                     verify_command, trajectory_factory=None, numeric_detours=False,all_joint_detours=False):
    """Same-state moving teacher proposal; never an execution/label receipt."""
    result=dict(kind='abstain',physics_executed=False,training_eligible=False,query=deepcopy(query))
    original=handoff.input_snapshot(trajectory);output=handoff.output_snapshot(trajectory)
    try:
        measured=_validate_query(query,reference);goal=contract._array(destination,(7,),'authenticated destination')
        plan=runtime.pose.bound_json(reference.snapshot()['sources']['fresh_plan'])
        canonical=np.asarray(plan.get('transfer_q'),float)
        need(canonical.ndim==2 and canonical.shape[1:]==(7,) and len(canonical)>=2 and
             np.isfinite(canonical).all() and any(np.array_equal(goal,q) for q in canonical[1:]),
             'Recovery destination is not an authenticated canonical waypoint')
        retained=np.asarray(original['fields']['current_position'])
        for q in (retained,measured,goal):verify_command(q)
        deadline=time.monotonic()+MAX_PLAN_CPU_S
        route,report=planner(measured.copy(),goal.copy(),deadline)
        need(route is not None, 'Recovery planner abstained: '+str(report))
        route=np.asarray(route,dtype=float)
        need(route.ndim==2 and route.shape[1:]==(7,) and len(route)>=2 and np.isfinite(route).all() and
             np.array_equal(route[0],measured) and np.array_equal(route[-1],goal), 'Recovery route endpoints differ')
        # Legacy connector is retained for v1. The v2 candidates use the actual
        # retained command seam and full preview, without stopping at sensor q.
        # Neither route resets q/v/a or substitutes measured command derivatives.
        candidates=(recovery_route_candidates(retained,route,all_joints=all_joint_detours) if numeric_detours else [np.vstack([retained,route])])
        attempts=[];preview=None
        for candidate in candidates:
            candidate=candidate[np.r_[True,np.max(abs(np.diff(candidate,axis=0)),axis=1)>1e-12]]
            if len(candidate)==1:candidate=np.vstack([candidate,candidate])
            if numeric_detours:
                try:
                    _preview(trajectory,command_history,candidate[1:],lambda q:None,settle=True,
                        trajectory_factory=trajectory_factory,deadline=deadline)
                except FinishedStationaryRoundoffError as error:
                    attempts.append(dict(numerical_proposal_accepted=False,reason=str(error),
                        details=deepcopy(getattr(error,'details',{})),physics_executed=False))
                    continue
            # Numeric preflight is not a safety check. All selected commands
            # still undergo the original full-mesh, posture and FD preview.
            preview=_preview(trajectory,command_history,candidate[1:],verify_command,settle=True,
                trajectory_factory=trajectory_factory,deadline=deadline)
            route=candidate;break
        result['numeric_proposal_rejections']=attempts
        need(preview is not None,'Every recovery numerical proposal was rejected')
        result.update(kind='recovery_proposal',route=route.tolist(),planner=report,preview=preview,
            destination_q=goal.tolist(),query_sha256=runtime.digest(query))
    except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
        result.update(kind='abstain',reason=type(error).__name__+': '+str(error),
            details=deepcopy(getattr(error,'details',{})))
    finally:
        need(handoff.input_snapshot(trajectory)==original and handoff.output_snapshot(trajectory)==output,
             'Same-state teacher changed live interpolator')
    return result


class GivenContactRecoveryExpert(GivenContactExpert):
    """One preview-safe target perturbation, then explicit measured-state replan."""
    def __init__(self,*args,recovery_binding,**kwargs):
        self.recovery_binding=recovery_binding;self._data_kind='nominal_teacher'
        self._recovery_events=[];self._recovery_queries=[];self._approach_cells=0
        self._perturbations=0;self._recovery_completed=False;self._attempted=False
        super().__init__(*args,**kwargs)

    def _issue_intent(self,target,context,role,*,stop=False):
        setter=super()._issue_intent(target,context,role,stop=stop)
        student_wait=(self._data_kind=='perturbation' and role=='given_l2_history_wait' and
            self.recovery_binding['settings']['schema']==STUDENT_SCHEMA)
        self._recovery_events.append(dict(intent_id=setter['intent_id'],setter_sha256=setter['setter_sha256'],
            physics_row_start=setter['physics_row_start'],physics_row_end_exclusive=setter['physics_row_start'],
            kind=self._data_kind if not stop and (role=='given_l2_motion' or student_wait) else 'excluded',completed=False))
        return setter

    def _advance_tick(self,subphase,**kwargs):
        try:return super()._advance_tick(subphase,**kwargs)
        finally:
            if self._recovery_events:
                event=self._recovery_events[-1];event['physics_row_end_exclusive']=len(self.env.rows)
                event['completed']=len(self.env.rows)-event['physics_row_start']==12

    def _tick(self,command,subphase,active=False,recovery=False):
        start=len(self.env.rows)
        try:return super()._tick(command,subphase,active=active,recovery=recovery)
        finally:
            raw=getattr(self,'_current_intent_audit',None)
            if len(self.env.rows)>start and raw is not None:
                need(bool(self._recovery_events),'Actual recovery setter lacks role ledger')
                entry=self._recovery_events[-1];setter=raw['setter']
                need(all(entry[k]==setter[k] for k in ('intent_id','setter_sha256','physics_row_start')),
                     'Actual native setter differs from role ledger')
                entry['physics_row_end_exclusive']=len(self.env.rows)
                entry['completed']=len(self.env.rows)-entry['physics_row_start']==12

    def _verify_recovery_command(self,q):
        need(self.posture_limits.evaluate(self.kin,self.env.base,q,contact=False)['clear'],
             'Recovery command original posture rejected')
        checked=self.checker.check(q,clearance=.003,allow_patient_contact=False,
            compression=max(0.,float(self.env.state()['compression_m'])))
        need(checked.get('clear') is True and np.isfinite(checked.get('minimum_distance_m',np.nan)) and
             checked['minimum_distance_m']>=.003-1e-9, 'Recovery command original collision rejected')

    def _query_state(self):
        observation=self._reference_observation();state=self.env.state();history,times=self._history.snapshot()
        return dict(measured_q=np.asarray(state['q']).tolist(),measured_dq=np.asarray(state['dq']).tolist(),
            history=history.tolist(),history_times=times.tolist(),time_s=float(state['time_s']),physics_row=len(self.env.rows),
            reference_sha256=self.reference.sha256,context=list(observation['context']),hall_N=float(state['hall_N']),
            physical_contact_N=float(sum(self.env.last_contacts.values())),compression_m=float(state['compression_m']),
            metadata=deepcopy(self.metadata))

    def _recover_to_destination(self,destination):
        settings=self.recovery_binding['settings'];student=settings['schema']==STUDENT_SCHEMA
        self._attempted=True;query=self._query_state();_validate_query(query,self.reference,replay_start=student)
        history=np.asarray([r['q_command'] for r in self.env.rows[-3:]],float)
        frozen=deepcopy(query);snapshot=handoff.input_snapshot(self.trajectory)
        if student:
            prefix=self.recovery_binding['student_prefix']
            need(query['physics_row']==prefix['start_row'] and
                query['context'][-1]==prefix['initial_context_phase'] and
                np.allclose(query['measured_q'],prefix['initial_measured_q'],atol=1e-5,rtol=0),
                'Fresh original home/history boundary differs from retained student source')
            targets=np.asarray(prefix['raw_targets'],float)
            roles=prefix['source_roles']
            provenance=dict(kind='student_prefix_preview',student_prefix=settings['student_prefix'],
                source_native_rows=[prefix['start_row'],prefix['end_row_exclusive']],
                source_policy_motion_cells=prefix['source_policy_motion_cells'],
                source_history_wait_cells=prefix['source_history_wait_cells'],
                learned_policy_executed=False,positive_teacher_labels=False)
            # A raw target is an intent, not an executed pose. The entire actual
            # retained-q/v/a command prefix receives the unchanged full checks.
        else:
            rng=np.random.default_rng(settings['seed'])
            offset=rng.normal(size=7);offset*=settings['amplitude_rad']/np.max(abs(offset))
            target=np.asarray(self.trajectory.input.target_position)+offset
            self._verify_recovery_command(target);targets=[target]*PERTURB_CELLS;roles=['given_l2_motion']*PERTURB_CELLS
            provenance=dict(kind='perturbation_preview',offset_rad=offset.tolist(),target_q=target.tolist())
        perturb=preview_cells(self.trajectory,history,targets,self._verify_recovery_command)
        need(self._query_state()==frozen and handoff.input_snapshot(self.trajectory)==snapshot,
             'Physics/history changed during perturbation preview')
        self._recovery_queries.append(dict(provenance,query=frozen,
            preview={k:v for k,v in perturb.items() if k not in ('commands','cells')}))
        self._data_kind='perturbation'
        for target,role in zip(targets,roles):
            observation=self._reference_observation()
            self._issue_intent(target,observation['context'],role)
            for _ in range(12):
                self._advance_tick(role);observation=self._reference_observation()
                if self._handoff_ready(observation):
                    need(not student,'Retained student prefix reached handoff before its declared cut')
                    return True
        self._perturbations+=1;self._data_kind='recovery_teacher'
        query=self._query_state();history=np.asarray([r['q_command'] for r in self.env.rows[-3:]],float)
        frozen=deepcopy(query);snapshot=handoff.input_snapshot(self.trajectory)
        proposal=propose_recovery(query,self.trajectory,history,destination,reference=self.reference,
            planner=lambda start,goal,deadline:self._plan_inter_point_bridge(start,goal,deadline),
            verify_command=self._verify_recovery_command,
            numeric_detours=settings['schema'] in (LATE_SCHEMA,STUDENT_SCHEMA),all_joint_detours=student)
        saved=deepcopy(proposal)
        if saved.get('preview'):saved['preview']={k:v for k,v in saved['preview'].items() if k not in ('commands','cells')}
        self._recovery_queries.append(saved)
        need(self._query_state()==frozen and handoff.input_snapshot(self.trajectory)==snapshot,
             'Physics/history changed during recovery planning')
        need(proposal['kind']=='recovery_proposal','Recovery teacher abstained: '+str(proposal.get('reason')))
        for waypoint in np.asarray(proposal['route'])[1:]:
            # Recheck the actual retained derivative seam after any real settling
            # delay at the previous waypoint; all motion uses original teacher.
            history=np.asarray([r['q_command'] for r in self.env.rows[-3:]],float)
            _preview(self.trajectory,history,[waypoint],self._verify_recovery_command,settle=True)
            if super()._teacher_waypoint(waypoint):
                self._recovery_completed=True;return True
        self._recovery_completed=True;self._data_kind='nominal_teacher'
        need(not student,'Student-state expert reached final waypoint without qualified handoff')
        return False

    def _teacher_waypoint(self,destination):
        for _ in range(120*240):
            observation=self._reference_observation()
            if self._interior_policy() and self._handoff_ready(observation):return True
            if len(self.env.rows)%12==0:
                if len(self._history.frames)!=2:
                    if self._wait_history_cell(observation)=='handoff':return True
                    continue
                student=self.recovery_binding['settings']['schema']==STUDENT_SCHEMA
                if (observation['phase']=='APPROACH' or student and observation['phase']=='FREE') and not self._attempted:
                    self._approach_cells+=1
                    pose=self.kin.tcp(np.asarray(self.env.state()['q']))
                    distance=float(np.linalg.norm(pose[:3,3]-self.reference.pose_base[:3,3]))
                    if recovery_trigger(self.recovery_binding['settings'],phase=observation['phase'],
                            approach_cells=self._approach_cells,distance_m=distance):
                        if self.recovery_binding['settings']['schema']==STUDENT_SCHEMA:
                            plan=runtime.pose.bound_json(self.reference.snapshot()['sources']['fresh_plan'])
                            destination=np.asarray(plan['transfer_q'],float)[-1]
                        return self._recover_to_destination(destination)
                if (np.max(abs(self.q-destination))<=1e-10 and np.max(abs(self.trajectory.velocity))<=1e-10 and
                        np.max(abs(self.trajectory.acceleration))<=1e-10 and self._stationary()):return False
                self._issue_intent(destination,observation['context'],'given_l2_motion')
            self._advance_tick(self._dispatcher().current['execution_role'])
        raise ValueError('Recovery teacher original L2 waypoint timeout')

    def run(self,points):
        result=None
        try:
            result=super().run(points)
            if not (self._perturbations==1 and self._recovery_completed):
                record_incomplete_recovery(result)
            return result
        finally:
            value=dict(schema=SIDECAR_SCHEMA,episode_id=self.episode_id,reference_sha256=self.reference.sha256,
                config_sha256=self.recovery_binding['config_sha256'],declaration=self.recovery_binding['settings'],
                completed=bool(result and result.get('success') is True and self._recovery_completed and self._perturbations==1),
                native_steps=len(self.env.rows),events=self._recovery_events,queries=self._recovery_queries,
                perturbations_executed=self._perturbations,training_eligible=False)
            ref=runtime.write_new(self.env.out/'given_recovery.json',value)
            if result is not None:result.update(recovery_collection=ref,expert_demonstration=False)


def make_controller(*args,config,**kwargs):
    binding=preflight(config);meta=binding['metadata'];ref=binding['reference']
    provider=GivenPoseProvider(ref,meta['episode_id'],meta)
    return GivenContactRecoveryExpert(*args,metadata=meta,episode_id=meta['episode_id'],estimate_provider=provider,
        recovery_binding=binding,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config',required=True,type=Path);parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--episode-id',required=True);parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--amplitude',required=True,type=float,choices=(.015,.03));parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();ref=runtime.reference(args.source_config)
    need(ref['sha256']==args.source_sha256,'Original teacher config digest differs')
    config=derive_config(ref,episode_id=args.episode_id,seed=args.seed,amplitude_rad=args.amplitude)
    runtime.write_new(args.output,config)


if __name__=='__main__':main()
