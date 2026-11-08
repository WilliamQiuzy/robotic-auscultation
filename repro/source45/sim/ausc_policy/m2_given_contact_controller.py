"""Separately owned bounded initialization for prescribed contact-pose control.

Teacher and ACT share first measured coarse entry and persistent parking.
ALIGN is analytic L3, with no patient-contact permission and no L2 fallback.
All execution still traverses FreshExpert's original native safety chain.
"""
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from . import m2_contract as contract, m2_stage as stage
from .m2_given_contact_pose import TASK_ID, GivenPoseProvider, require_route, need
from .m2_given_contact_intent import IntentDispatcher, ACTION_SEMANTICS_ID
from .m2_act_controller import M2ACTController, LIMITS, approach_gate
from .fresh_expert import FreshExpert, contact_plane_contract, contact_plane_observation
from .contact_entry import check_retained_pose
from .retainer_contact import ALL_FORCE_GROUPS
from . import m2_given_contact_handoff as handoff
from sim.auscultation_expert_layout_v1.controller import DT, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from sim.auscultation_pi05_v6.execution import JointTrajectory,FinishedStationaryRoundoffError
from sim.auscultation_skill_v1.executor import command_derivatives

BOUNDS = dict(translation_m=.003, rotation_rad=float(np.deg2rad(5.)),
              simulation_s=10., planning_cpu_s=30.)

# A target at the exact hard ceiling can acquire a few positive ULPs when
# adjacent float64 position commands are divided by DT. Plan conservatively;
# NativeVelocityFeedforward's exact .175 rejection remains unchanged.
GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S = 1e-6
GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3 = handoff.L2_JERK_NUMERICAL_RESERVE_RAD_S3
GIVEN_TRAJECTORY_LIMITS = dict(LIMITS,
    velocity=LIMITS['velocity']-GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S,
    jerk=LIMITS['jerk']-GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3)
ARCHIVAL_GIVEN_TRAJECTORY_LIMITS = dict(LIMITS,velocity=LIMITS['velocity']-GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S)


def _bounds(value):
    value = BOUNDS if value is None else value
    need(value == BOUNDS, 'Frozen ALIGN bounds cannot be enlarged or replaced')
    return dict(value)


def _metrics(pose, reference):
    pose = contract._pose(pose, 'physical pose')
    delta = pose[:3, 3]-reference.pose_base[:3, 3]; normal = reference.normal_base
    gap = float(delta @ normal)
    return dict(gap_m=gap, tangent_m=float(np.linalg.norm(delta-normal*gap)),
        angle_deg=float(np.rad2deg(np.arccos(np.clip(pose[:3, 2] @ -normal, -1., 1.)))))


def _plane(pose, plane, base):
    need(plane is not None and plane.get('plane_tilt_limit_deg') == 3. and plane.get('tilt_relaxed') is False,
         'Original 3 degree local plane is mandatory during ALIGN')
    result = contact_plane_observation(plane, np.asarray(base)[:3, :3] @ pose[:3, 2])
    need(result['passed'], 'ALIGN local plane rejected')
    return result


def alignment_admission(retained_pose, given_reference, original_gates, bounds=None):
    """Additional full-R/plane gates never replace original measured entry."""
    b = _bounds(bounds); g = original_gates; ref = given_reference
    need(g['reference_sha256'] == ref.sha256, 'ALIGN reference revision changed')
    need(not g['original_violations'], 'Original force/stroke/posture safety gate failed')
    forces = np.asarray(list(g['patient_contacts'].values()), float)
    need(np.isfinite(forces).all() and np.all(forces == 0.), 'ALIGN forbids patient/forbidden contact')
    result = dict(passed=True)
    for name, pose, speed in [('command', retained_pose, 0.),
                              ('observed', g['measured_pose_base'], g['measured_speed_m_s'])]:
        pose = contract._pose(pose, name+' ALIGN pose')
        coarse = approach_gate(dict(_metrics(pose, ref), tcp_speed_m_s=speed), g['phase_config'])
        angle = float(Rotation.from_matrix(pose[:3, :3] @ ref.pose_base[:3, :3].T).magnitude())
        need(coarse['passed'], 'ALIGN admission outside original coarse entry: '+name)
        need(angle <= b['rotation_rad']+1e-12, 'ALIGN full rotation admission rejected: '+name)
        plane = _plane(pose, g['plane'], g['world_from_base'])
        result[name] = dict(coarse=coarse, full_rotation_error_rad=angle, plane=plane)
    return result


class AlignmentAudit:
    """Independent command/observed displacement, path and full SO(3) audits."""
    def __init__(self, reference, command_pose, observed_pose, plane, base, bounds=None):
        self.bounds = _bounds(bounds); self.reference = reference
        self.plane, self.base = deepcopy(plane), np.asarray(base).copy()
        self.origins = {}; self.previous = {}; self.values = {}
        for kind, pose in [('command', command_pose), ('observed', observed_pose)]:
            pose = contract._pose(pose, kind+' ALIGN origin')
            self.origins[kind] = pose; self.previous[kind] = pose
            self.values[kind] = dict(max_translation_m=0., path_length_m=0.,
                max_full_rotation_rad=0., rotation_path_rad=0., max_local_plane_deg=0., samples=0)
            self.observe(pose, kind)

    def observe(self, pose, kind):
        need(kind in self.values, 'Unknown ALIGN audit channel')
        pose = contract._pose(pose, kind+' ALIGN sample')
        plane = _plane(pose, self.plane, self.base)
        origin, previous, row = self.origins[kind], self.previous[kind], self.values[kind]
        row['path_length_m'] += float(np.linalg.norm(pose[:3, 3]-previous[:3, 3]))
        row['max_translation_m'] = max(row['max_translation_m'], float(np.linalg.norm(pose[:3, 3]-origin[:3, 3])))
        row['rotation_path_rad'] += float(Rotation.from_matrix(pose[:3, :3] @ previous[:3, :3].T).magnitude())
        row['max_full_rotation_rad'] = max(row['max_full_rotation_rad'],
            float(Rotation.from_matrix(pose[:3, :3] @ origin[:3, :3].T).magnitude()))
        row['max_local_plane_deg'] = max(row['max_local_plane_deg'], plane['tilt_deg'])
        row['samples'] += 1
        self.previous[kind] = pose.copy()
        need(max(row['max_translation_m'], row['path_length_m']) <= self.bounds['translation_m']+1e-12,
             'ALIGN '+kind+' translation/path budget rejected')
        need(max(row['max_full_rotation_rad'], row['rotation_path_rad']) <= self.bounds['rotation_rad']+1e-12,
             'ALIGN '+kind+' full rotation budget rejected')
        return deepcopy(row)

    def report(self):
        return deepcopy(self.values)


@dataclass(frozen=True)
class AlignmentProposal:
    commands: np.ndarray
    report: dict

    def __post_init__(self):
        values = np.asarray(self.commands, dtype=np.float64)
        object.__setattr__(self, 'commands', np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape))


def propose_l3_alignment(retained_q, measured_q, given_reference, original_gates, bounds=None,
                         *, solver=None, started_at_s=None):
    """Small fixed-normal-coordinate line + shortest full rotation, fail closed.

IK is L3-only. A 13-knot Cartesian proposal is checked at every actual 240 Hz
command after retiming, including its final stationary stop. No detour/retry
with enlarged geometry, changed origin, relaxed clearance or alternate R.
"""
    b = _bounds(bounds); g = original_gates; ref = given_reference
    clock = g.get('clock', time.monotonic)
    started = clock() if started_at_s is None else started_at_s
    deadline = started+b['planning_cpu_s']
    def budget():
        need(clock() < deadline, 'ALIGN planning/complete verification CPU deadline')
    budget()
    q0 = contract._array(retained_q, (7,), 'retained q0')
    contract._array(measured_q, (7,), 'measured q0')
    history = contract._array(g['command_history'], (3, 7), 'real native command history')
    need(np.array_equal(history, np.tile(q0, (3, 1))), 'ALIGN requires real parked command history')
    kin = g['kin']; start = kin.tcp(q0)
    admission = alignment_admission(start, ref, g, b)
    normal = ref.normal_base; target = ref.pose_base[:3, 3]
    gap = float((start[:3, 3]-target) @ normal)
    goal = target+normal*gap
    rotvec = Rotation.from_matrix(ref.pose_base[:3, :3] @ start[:3, :3].T).as_rotvec()
    need(np.linalg.norm(goal-start[:3, 3]) <= b['translation_m'] and np.linalg.norm(rotvec) <= b['rotation_rad'],
         'ALIGN correction cap rejected')
    if solver is None:
        from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik
        solver = solve_posture_ik
    parameters = np.linspace(0., 1., 13); knots = [q0]
    for u in parameters[1:]:
        budget()
        rotation = Rotation.from_rotvec(u*rotvec).as_matrix() @ start[:3, :3]
        # Reserve the soft joint box for cubic undershoot between valid IK knots.
        q, accepted, _, _ = solver(kin, g['base'], g['posture_limits'],
            start[:3, 3]+u*(goal-start[:3, 3]), rotation, knots[-1], deadline=deadline,
            planning_margin_m=.002, joint_margin_rad=1e-4, contact=False, maxiter=80)
        need(accepted, 'ALIGN full-pose IK rejected')
        knots.append(contract._array(q, (7,), 'ALIGN knot'))
    spline = CubicSpline(parameters, np.asarray(knots), axis=0)
    # Retiming changes time only. The Cartesian line/full-R path is never enlarged.
    duration = 1.
    for _ in range(16):
        budget(); count = int(np.ceil(duration/DT))
        need((count+3)*DT <= b['simulation_s']+1e-12, 'ALIGN simulation budget including final stop')
        t = np.arange(1, count+1)/count; u = 10*t**3-15*t**4+6*t**5
        commands = spline(u); commands[-1] = knots[-1]
        commands = np.vstack([commands, np.tile(commands[-1], (3, 1))])
        values = np.vstack([history, commands])
        maxima = dict(velocity=0., acceleration=0., jerk=0.)
        for order, name in [(1, 'velocity'), (2, 'acceleration'), (3, 'jerk')]:
            maxima[name] = float(np.max(np.abs(np.diff(values, n=order, axis=0)))/DT**order)
        ratio = max(maxima['velocity']/.175, np.sqrt(maxima['acceleration']/.5), np.cbrt(maxima['jerk']/1.5))
        if ratio <= 1.+1e-10:
            break
        duration = count*DT*max(1.05, ratio*1.01)
    else:
        raise ValueError('ALIGN derivative retiming exhausted')
    audit = AlignmentAudit(ref, start, g['measured_pose_base'], g['plane'], g['world_from_base'], b)
    for index, command in enumerate(commands):
        budget()
        need(np.all(command >= kin.lower_limits) and np.all(command <= kin.upper_limits), 'ALIGN joint limit rejected')
        need(g['posture_limits'].evaluate(kin, g['base'], command, contact=False)['clear'], 'ALIGN posture rejected')
        checked = g['checker'].check(command, clearance=.003, allow_patient_contact=False,
                                     compression=max(0., g['compression_m']))
        need(checked.get('clear') is True and np.isfinite(checked.get('minimum_distance_m', np.nan)) and
             checked['minimum_distance_m'] >= .003-1e-9, 'ALIGN noncontact collision/clearance rejected')
        pose = kin.tcp(command); audit.observe(pose, 'command')
        planned_plane = _plane(pose, g['plane'], g['world_from_base'])
        need(planned_plane['tilt_deg'] <= 2.75+1e-9, 'ALIGN original .25 degree planning plane reserve')
        progress = float(u[index]) if index < count else 1.
        desired_p = start[:3, 3]+progress*(goal-start[:3, 3])
        desired_r = Rotation.from_rotvec(progress*rotvec).as_matrix() @ start[:3, :3]
        need(np.linalg.norm(pose[:3, 3]-desired_p) <= 2e-5 and
             Rotation.from_matrix(pose[:3, :3] @ desired_r.T).magnitude() <= 2e-4,
             'ALIGN deviates from fixed-normal-coordinate line/shortest rotation')
    budget()
    return AlignmentProposal(commands, dict(schema='m2_given_contact_l3_align_v1', admission=admission,
        frozen_q0=q0.tolist(), frozen_measured_q=np.asarray(measured_q).tolist(), reference_sha256=ref.sha256,
        frozen_measured_pose_base=np.asarray(g['measured_pose_base']).tolist(),
        frozen_command_history=history.tolist(), reference_revision=0,
        frozen_command_velocity=((history[-1]-history[-2])/DT).tolist(),
        frozen_command_acceleration=((history[-1]-2*history[-2]+history[-3])/DT**2).tolist(),
        frozen_gap_m=gap, bounds=b, duration_s=len(commands)*DT, planning_cpu_s=clock()-started,
        all_240hz_commands_verified=True, final_stationary_stop_included=True,
        command_derivative_maxima=maxima, preflight_audit=audit.report(),
        physical_phase='APPROACH', patient_contact_permitted=False))


class GivenContactBoundary:
    """Shared physical boundary and L3 initialization; no expert L2 fallback."""
    task_id = TASK_ID

    def _interior_policy(self):
        policy=getattr(self,'metadata',{}).get('handoff_policy_id')
        need(policy in (None,handoff.POLICY_ID), 'Unknown given-contact handoff policy')
        return policy==handoff.POLICY_ID

    def _trajectory_limits(self):
        return dict(GIVEN_TRAJECTORY_LIMITS if self._interior_policy() else ARCHIVAL_GIVEN_TRAJECTORY_LIMITS)

    def _actual_parking_pose(self):
        need(bool(self.env.rows), 'Actual native parking speed/history required')
        sample=handoff.measured_sample(self.env.state(),self.env.base,len(self.env.rows),
            self.env.rows[-1].get('tcp_speed_m_s'))
        capture=getattr(self.env,'raw_capture',None)
        if self.control_owner=='L2_STOP' and getattr(capture,'metadata',{}).get('handoff_policy_id')==handoff.POLICY_ID:
            recorded=getattr(capture,'_last_parking_pose',None)
            need(recorded==sample, 'Online measured parking pose differs from POST-step native capture')
            return deepcopy(recorded)
        return sample

    def _handoff_ready(self, observation):
        coarse=self._handoff_measurements()
        if not self._interior_policy():return coarse['passed']
        need(self.l2_handoffs and self.control_owner in ('TEACHER_L2','ACT_L2'), 'Interior observation requires actual L2 owner')
        state=self.env.state();stamp=float(state['time_s']);row=len(self.env.rows)
        context=observation.get('context')
        need(isinstance(context,(tuple,list)) and len(context)==4 and
             tuple(context[:3])==(self.episode_id,self.target.name,0) and context[3] in ('FREE','APPROACH'),
             'Interior handoff episode/reference context changed')
        need(abs(observation['observed_at_s']-stamp)<=1e-8 and
             require_route(self.metadata).sha256==self.reference.sha256, 'Stale or changed interior reference observation')
        receipt=self.l2_handoffs[-1]
        rejections=receipt.get('parking_numeric_rejections',[])
        if rejections:
            need(row-receipt['parking_budget_start_physics_row']<handoff.MAX_PARK_STEPS,
                 'Given-contact numerical preview deferral exceeded single 5 second parking budget')
            if row==rejections[-1]['physics_row']:return False
        receipt.setdefault('handoff_policy_id',handoff.POLICY_ID)
        if coarse['passed'] and 'coarse_entry' not in receipt:
            receipt.update(l2_coarse_reached=True,coarse_entry=dict(deepcopy(coarse),
                reference_sha256=self.reference.sha256,reference_revision=0,context=list(observation['context'])))
        if row%12:return False
        sample=self._actual_parking_pose()
        trigger=handoff.interior_trigger(self.reference,command_pose=self.kin.tcp(self.q),
            observed_pose=sample['pose_base'],measured_speed_m_s=sample['tcp_speed_m_s'],
            phase_config=self.manager.config,physics_row=row,time_s=stamp,context=observation['context'],
            reference_sha256=self.reference.sha256)
        if trigger['passed']:
            need(coarse['passed'] and 'coarse_entry' in receipt, 'Interior trigger cannot replace first original coarse entry')
            receipt['parking_trigger']=trigger
        return trigger['passed']

    def _verify_parking_command(self, command):
        pose=self.kin.tcp(command)
        need(approach_gate(dict(handoff.pose_metrics(pose,self.reference),tcp_speed_m_s=0.),self.manager.config)['passed'],
             'Preview/actual parking command left original coarse tube')
        need(self.posture_limits.evaluate(self.kin,self.env.base,command,contact=False)['clear'],
             'Preview/actual parking original posture rejected')
        checked=self.checker.check(command,clearance=.003,allow_patient_contact=False,
            compression=max(0.,float(self.env.state()['compression_m'])))
        need(checked.get('clear') is True and np.isfinite(checked.get('minimum_distance_m',np.nan)) and
             checked['minimum_distance_m']>=.003-1e-9, 'Preview/actual parking original collision clearance rejected')

    def _parking_live_snapshot(self):
        commands=np.asarray([row['q_command'] for row in self.env.rows],dtype=np.float64)
        import hashlib
        history=getattr(self._history,'frames',[])
        return dict(input=handoff.input_snapshot(self.trajectory),command_q=self.q.tolist(),
            interpolator_output=handoff.output_snapshot(self.trajectory),
            interpolator_object_ids={name:id(value) for name,value in dict(trajectory=self.trajectory,
                input=self.trajectory.input,output=self.trajectory.output,generator=self.trajectory.generator).items()},
            native_commands_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
            command_history=np.asarray(commands[-3:]).tolist(),rows=len(self.env.rows),
            measured_pose=self._actual_parking_pose(),reference_sha256=require_route(self.metadata).sha256,
            causal_history=[(float(t),np.asarray(q).tolist()) for t,q in history],
            prefix=(None if self.prefix is None else dict(index=self.prefix.index,
                absolute=self.prefix.absolute.tolist(),context=list(self.prefix.context))))

    def _stop_for_interior_handoff(self, context):
        receipt=self.l2_handoffs[-1];trigger=deepcopy(receipt.get('parking_trigger'))
        need(trigger and trigger['passed'] and trigger['context']==list(context) and
             trigger['physics_row']==len(self.env.rows), 'Fresh interior parking trigger required before preview')
        observation=self._reference_observation()
        need(self._handoff_ready(observation) and receipt['parking_trigger']==trigger, 'Interior trigger changed before preview')
        self.reference.verify_sources()
        history=np.asarray([row['q_command'] for row in self.env.rows[-3:]],float)
        need(history.shape==(3,7) and np.array_equal(history[-1],self.q), 'Actual current q/history differs before preview')
        started=time.monotonic();prior_cpu=sum(r['planning_cpu_s'] for r in receipt.get('parking_numeric_rejections',[]))
        live_before=self._parking_live_snapshot()
        try:
            preview=handoff.preview_stop(self.trajectory,history,self._verify_parking_command,self._parking_live_snapshot,
                started_at_s=started-prior_cpu)
        except FinishedStationaryRoundoffError as error:
            # Reject only this read-only numerical proposal. Live execution,
            # geometric/FD/source gates and all other exceptions stay fatal.
            need(self._parking_live_snapshot()==live_before,'Numerically rejected STOP preview mutated live state')
            cost=time.monotonic()-started
            need(prior_cpu+cost<handoff.MAX_PREVIEW_CPU_S,'Cumulative stop preview CPU deadline')
            receipt.setdefault('parking_budget_start_physics_row',len(self.env.rows))
            receipt.setdefault('parking_numeric_rejections',[]).append(dict(
                schema='m2_given_stop_preview_numeric_rejection_v1',exception_type=type(error).__name__,
                failure_reason=str(error),details=deepcopy(getattr(error,'details',{})),
                physics_row=len(self.env.rows),time_s=float(self.env.state()['time_s']),planning_cpu_s=cost,
                input_state=live_before['input'],input_sha256=handoff.digest(live_before['input']),
                frozen_command_history=history.tolist(),live_state_before=live_before,live_state_after=self._parking_live_snapshot(),
                stop_setter_issued=False,physics_advanced=False,live_state_unchanged=True))
            receipt.pop('parking_trigger',None)
            return None
        except Exception as error:
            receipt['parking_preview_failure']=dict(schema=handoff.PREVIEW_SCHEMA,handoff_policy_id=handoff.POLICY_ID,
                passed=False,failure_reason=str(error),physics_row=len(self.env.rows),time_s=float(self.env.state()['time_s']),
                planning_cpu_s=time.monotonic()-started,max_stop_steps=handoff.MAX_STOP_STEPS,
                reserved_settle_intervals=handoff.SETTLE_INTERVALS,max_parking_steps=handoff.MAX_PARK_STEPS,
                stop_setter_issued=False)
            if hasattr(error,'stop_residual_report'):
                receipt['parking_preview_failure']['stop_residual_adapter']=deepcopy(error.stop_residual_report)
            raise
        import inspect,hashlib
        from pathlib import Path
        preview.report['source_sha256']={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (handoff.__file__,inspect.getfile(type(self.trajectory)))}
        if 'stop_residual_adapter' in preview.report:
            preview.report['source_sha256'].update(preview.report['stop_residual_adapter']['source_sha256'])
        self.reference.verify_sources()
        preview.report['planning_cpu_s']=prior_cpu+time.monotonic()-started
        need(preview.report['planning_cpu_s']<handoff.MAX_PREVIEW_CPU_S, 'Complete stop preview/source verification CPU deadline')
        budget_start=receipt.get('parking_budget_start_physics_row',len(self.env.rows))
        remaining=handoff.MAX_PARK_STEPS-(len(self.env.rows)-budget_start)
        need(len(preview.commands)+handoff.SETTLE_INTERVALS<=remaining,
             'Verified STOP/settle cannot fit remaining single 5 second parking budget')
        # No setter, actor transition or live q/v/a mutation occurs before a
        # complete safe preview has passed its CPU and seam verification.
        receipt['parking_preview']=deepcopy(preview.report)
        transition=handoff.apply_stop_profile(self.trajectory,expected_input=preview.report['input_state'])
        need(transition==preview.report['stop_profile_transition'], 'Actual STOP limit transition differs from verified clone')
        receipt['parking_stop_profile_transition']=transition
        self.prefix=None;self._history=None;self.control_owner='L2_STOP'
        self._issue_intent(self.q.copy(),context,'given_l2_park',stop=True)
        stop_input=handoff.input_snapshot(self.trajectory,profile='stop')
        need(stop_input==preview.report['stop_input_state'], 'Actual STOP input differs from verified clone after its setter')
        receipt.update(parking_stop_input_state=stop_input,parking_stop_input_sha256=handoff.digest(stop_input))
        if 'stop_residual_adapter' in preview.report:
            self.trajectory.begin_stop_residual_replanning(history=history)
        start=len(self.env.rows);settle=[]
        for i in range(remaining):
            if i<len(preview.commands):
                self._advance_tick('given_l2_park')
            else:
                GivenContactBoundary._advance_tick(self,'given_l2_park',hold_command=preview.commands[-1])
            need(self._reference_observation()['context']==context, 'Reference context changed during parking')
            expected=preview.commands[min(i,len(preview.commands)-1)]
            need(np.array_equal(self.q,expected), 'Actual parking command differs bit-exactly from clone preview')
            self._verify_parking_command(self.q)
            entry=self._handoff_measurements()
            need(entry['passed'], 'Original coarse entry lost during bounded parking')
            if i+1>=len(preview.commands):
                need(np.max(abs(self.trajectory.velocity))<=1e-10 and
                     np.max(abs(self.trajectory.acceleration))<=1e-10 and self._stationary(),
                     'Actual stop stationarity differs from preview')
                receipt.setdefault('parking_stop_complete_physics_row',len(self.env.rows))
                if 'stop_residual_adapter' in preview.report and 'parking_stop_residual_adapter' not in receipt:
                    from . import m2_stop_residual_trajectory as residual
                    need(self.trajectory.stop_residual_terminal_ready(),
                         'Actual residual STOP terminal native readback is not certified')
                    actual=self.trajectory.stop_residual_report();residual.verify_report(actual,preview.commands)
                    need(actual['events']==preview.report['stop_residual_adapter']['events'] and
                         actual['events_sha256']==preview.report['stop_residual_adapter']['events_sha256'],
                         'Actual STOP cache-recalculation evidence differs from preview')
                    receipt['parking_stop_residual_adapter']=actual
                if 'parking_tail' not in receipt:
                    receipt['parking_tail']=handoff.constant_tail_snapshot(self.trajectory,self.q,len(self.env.rows))
                    self._parking_tail_objects=(self.trajectory,self.trajectory.input,self.trajectory.output,self.trajectory.generator)
                    self._parking_tail_audit=handoff.constant_tail_binding(receipt['parking_tail'])
                settle.append(self._actual_parking_pose());settle=settle[-25:]
                checked=handoff.settle_window(settle)
                if checked['passed']:
                    receipt.update(parking_settle=checked,parking_start_physics_row=start,
                        parking_end_physics_row=len(self.env.rows),parking_steps=len(self.env.rows)-start,
                        parking_budget_start_physics_row=budget_start,parking_budget_steps=len(self.env.rows)-budget_start,
                        handoff_policy_id=handoff.POLICY_ID)
                    self.control_owner='L3_ENTRY'
                    return entry
        raise EpisodeRejected('Given-contact interior stop/actual settling exceeded single 5 second parking budget')

    def _reference_observation(self):
        state = self.env.state()
        record = self.estimate_provider(self, state)
        self.actor_phase = record['phase']
        result = dict(record, context=(record['episode_id'], record['target_id'], record['revision'], record['phase']))
        if self.control_owner == 'TEACHER_L2' and isinstance(self._history, stage.CausalHistory):
            state17 = contract.encode_state(record['measured_q'],record['estimated_target_base'],
                                            record['confidence'],metadata=self.metadata)
            self._history.observe(state17,record['observed_at_s'],result['context'])
        return result

    def _handoff_measurements(self):
        state = self.env.state()
        speed = self.env.rows[-1].get('tcp_speed_m_s') if self.env.rows else None
        if speed is None:
            return dict(passed=False, reason='No actual native TCP speed')
        gate = approach_gate(dict(self.metrics(state), tcp_speed_m_s=speed), self.manager.config)
        return dict(passed=bool(self.actor_phase == 'APPROACH' and gate['passed']), measured=gate,
            time_s=float(state['time_s']), physics_row=len(self.env.rows), command_q=self.q.tolist(),
            measured_q=np.asarray(state['q']).tolist(), retained_pose_is_not_l2_success_gate=True)

    def _stationary(self):
        return M2ACTController._stationary(self)

    def _deadline(self):
        return M2ACTController._deadline(self)

    def _advance_tick(self, subphase, *, hold_command=None):
        self._deadline(); stamp = float(self.env.state()['time_s']); count = len(self.env.rows)
        if self._interior_policy() and self.control_owner=='L2_STOP':
            need(subphase=='given_l2_park', 'STOP profile cannot own ordinary L2 actions')
            handoff.input_snapshot(self.trajectory,profile='stop')
        elif self._interior_policy() and self.control_owner in ('TEACHER_L2','ACT_L2'):
            handoff.input_snapshot(self.trajectory,profile='l2')
        handoffs=getattr(self,'l2_handoffs',None)
        residual_preview=(handoffs[-1].get('parking_preview',{}).get('stop_residual_adapter')
                          if self.control_owner=='L2_STOP' and handoffs else None)
        residual_configured=any(callable(getattr(value,'begin_stop_residual_replanning',None))
            for value in (self.trajectory,getattr(self,'trajectory_factory',None)))
        if self.control_owner=='L2_STOP' and residual_configured:
            need(isinstance(residual_preview,dict) and isinstance(residual_preview.get('events'),list) and
                 residual_preview['events'], 'Configured residual STOP requires its verified handoff preview')
        tail=None
        if hold_command is not None:
            need(self._interior_policy() and self.control_owner=='L2_STOP' and subphase=='given_l2_park' and
                 handoffs and 'parking_tail' in handoffs[-1],
                 'Verified constant tail is forbidden for ordinary L2/history-wait actions')
            receipt=handoffs[-1];tail=receipt['parking_tail']
            if 'stop_residual_adapter' in receipt.get('parking_preview',{}):
                need(self.trajectory.stop_residual_terminal_ready(),
                     'Actual residual STOP terminal native readback changed during constant tail')
            need(receipt.get('parking_stop_complete_physics_row')==tail['start_physics_row']<=count and
                 self._stationary() and np.max(abs(self.trajectory.velocity))<=1e-10 and
                 np.max(abs(self.trajectory.acceleration))<=1e-10 and
                 all(a is b for a,b in zip(self._parking_tail_objects,
                     (self.trajectory,self.trajectory.input,self.trajectory.output,self.trajectory.generator))) and
                 np.array_equal(hold_command,self.q) and np.array_equal(hold_command,tail['command_q']) and
                 handoff.constant_tail_snapshot(self.trajectory,self.q,tail['start_physics_row'])==tail,
                 'Verified constant tail state/object/command changed')
        dispatcher = self._dispatcher()
        evidence = dispatcher.snapshot(time_s=stamp,physics_row=count)
        if tail is not None:
            need(evidence['setter']['offgrid_stop'] is True and evidence['setter']['control_owner']=='L2_STOP',
                 'Constant tail must retain the original actual STOP setter/owner')
        # Ruckig's terminal residuals can drift at later step() calls. Once
        # the preview stop and real three-command stationarity pass, repeat
        # that verified command through the original native chain. The same
        # interpolator input/output/generator and nonzero residual q/v/a stay
        # untouched; this is an explicit constant tail, not continued Ruckig.
        command = self.trajectory.step() if tail is None else np.asarray(hold_command,float).copy()
        residual_evidence=None
        if tail is None and self.control_owner=='L2_STOP' and handoffs:
            preview=residual_preview
            if preview is not None:
                event=self.trajectory.last_stop_residual_event();tick=event['tick']
                need(1<=tick<=len(preview['events']) and event==preview['events'][tick-1],
                     'Actual STOP residual event differs from verified preview before physics')
                residual_evidence=dict(adapter_id=preview['adapter_id'],event=event,event_sha256=handoff.digest(event),
                    objects=self.trajectory.last_stop_residual_object_check())
        evidence['live_target_after_step_q'] = dispatcher.snapshot(time_s=stamp,physics_row=count)['live_target_q']
        self._current_intent_audit = evidence
        self._current_stop_residual_audit=residual_evidence
        try:
            self._block(np.asarray(command)[None],subphase,active=False)
            need(len(self.env.rows) == count+1 and abs(self.env.state()['time_s']-stamp-DT) <= 1e-8,
                 'Nonconsecutive given native physics clock/rows')
            need(np.array_equal(self.env.rows[-1]['q_command'],command), 'Given native command differs from interpolator')
            if tail is not None:
                need(dispatcher.current==evidence['setter'] and
                     handoff.constant_tail_snapshot(self.trajectory,self.q,tail['start_physics_row'])==tail,
                     'Actual constant tail mutated live input/output/STOP target or owner')
        finally:
            self._current_intent_audit = None
            self._current_stop_residual_audit=None

    def _dispatcher(self):
        if getattr(self,'_intent_dispatcher',None) is None:
            self._intent_dispatcher = IntentDispatcher(self.trajectory,reference_sha256=self.reference.sha256)
        need(self._intent_dispatcher.trajectory is self.trajectory, 'Given interpolator replaced after intent start')
        return self._intent_dispatcher

    def _issue_intent(self, target, context, role, *, stop=False):
        return self._dispatcher().issue(target,time_s=float(self.env.state()['time_s']),
            physics_row=len(self.env.rows),context=context,owner=self.control_owner,role=role,stop=stop)

    def _wait_history_cell(self, observation):
        """Hold live intent; only the next declared grid may refresh its setter."""
        if self._interior_policy() and self._handoff_ready(observation):return 'handoff'
        if len(self.env.rows)%12 == 0:
            self._issue_intent(np.asarray(self.trajectory.input.target_position),
                               observation['context'],'given_l2_history_wait')
        while True:
            current = self._dispatcher().current
            self._advance_tick(current['execution_role'])
            observation = (self._observation() if isinstance(self,M2ACTController) else self._reference_observation())
            if self._handoff_ready(observation): return 'handoff'
            if len(self.env.rows)%12 == 0: return 'continue'

    def _stop_for_handoff(self, context):
        if self._interior_policy():return self._stop_for_interior_handoff(context)
        coarse = self._handoff_measurements()
        need(coarse['passed'], 'Original coarse entry lost before bounded parking')
        if self.l2_handoffs:
            self.l2_handoffs[-1].update(l2_coarse_reached=True, coarse_entry=deepcopy(coarse))
        self.prefix = None; self._history = None; self.control_owner = 'L2_STOP'
        # Retain the running q/v/a state; never recreate/zero the interpolator.
        self._issue_intent(self.q.copy(),context,'given_l2_park',stop=True)
        for _ in range(5*240):
            self._advance_tick('given_l2_park')
            need(self._reference_observation()['context'] == context, 'Reference context changed during parking')
            entry = self._handoff_measurements()
            need(entry['passed'], 'Original coarse entry lost during bounded parking')
            if (np.max(abs(self.trajectory.velocity)) <= 1e-10 and
                    np.max(abs(self.trajectory.acceleration)) <= 1e-10 and self._stationary()):
                self.control_owner = 'L3_ENTRY'
                return entry
        raise EpisodeRejected('Given-contact L2 parking timeout')

    def _alignment_gates(self):
        state = self.env.state(); observed = np.eye(4)
        observed[:3, :3] = self.env.base[:3, :3].T @ state['rotation_world']
        observed[:3, 3] = state['tcp_base']
        violations, _ = self._violations(state)
        for q, observed_state in [(self.q, False), (np.asarray(state['q']), True)]:
            posture = self.posture_limits.evaluate(self.kin, self.env.base, q, observed=observed_state, contact=False)
            check = (self.checker.check_observed(q, contact=False, clearance=.003,
                allow_patient_contact=False, compression=max(0., float(state['compression_m']))) if observed_state else
                self.checker.check(q, clearance=.003, allow_patient_contact=False,
                    compression=max(0., float(state['compression_m']))))
            if (not posture['clear'] or check.get('clear') is not True or
                    not np.isfinite(check.get('minimum_distance_m', np.nan)) or check['minimum_distance_m'] < .003-1e-9):
                violations.append('ALIGN command/observed posture/collision admission')
        self.static_world.check()
        return dict(kin=self.kin, base=self.env.base, world_from_base=self.env.base,
            phase_config=self.manager.config, plane=contact_plane_contract(self._point_plan),
            posture_limits=self.posture_limits, checker=self.checker, measured_pose_base=observed,
            measured_speed_m_s=self.env.rows[-1]['tcp_speed_m_s'], original_violations=violations,
            patient_contacts={key:self.env.last_contacts.get(key, 0.) for key in ALL_FORCE_GROUPS},
            reference_sha256=require_route(self.metadata).sha256, compression_m=float(state['compression_m']),
            command_history=np.asarray([row['q_command'] for row in self.env.rows[-3:]], float))

    def _contact_cycle(self):
        need(self.l2_handoffs and self.l2_handoffs[-1].get('l2_handoff_success') is True and
             self.control_owner == 'L3_ENTRY' and self._stationary() and self._handoff_measurements()['passed'],
             'No verified parked L2 entry for ALIGN')
        receipt = self.l2_handoffs[-1]; self.prefix = None; self._history = None
        self.reference.verify_sources()
        need(self.manager.state.phase == Phase.APPROACH and not self._contact_active, 'ALIGN requires noncontact APPROACH')
        gates = self._alignment_gates()
        admission = alignment_admission(self.kin.tcp(self.q), self.reference, gates)
        receipt.update(alignment_admission=admission, alignment_success=False, l3_started=False)
        self.control_owner = 'ANALYTIC_L3_ALIGN'
        state = self.env.state(); start_rows = len(self.env.rows); start_s = float(state['time_s'])
        q0 = self.q.copy(); measured = np.asarray(state['q']).copy()
        self._alignment_audit = AlignmentAudit(self.reference, self.kin.tcp(q0), gates['measured_pose_base'],
                                                gates['plane'], self.env.base)
        try:
            proposal = propose_l3_alignment(q0, measured, self.reference, gates)
            self.reference.verify_sources(); self.static_world.check()
            need(len(self.env.rows) == start_rows and self.env.state()['time_s'] == start_s and
                 np.array_equal(self.env.state()['q'], measured) and np.array_equal(self.q, q0),
                 'Physics/reference changed during ALIGN planning')
            receipt['alignment_plan'] = proposal.report
            for command in proposal.commands:
                need(self.env.state()['time_s']-start_s+DT <= BOUNDS['simulation_s']+1e-9,
                     'ALIGN simulation deadline includes final stop')
                self._block(command[None], 'L3_ALIGN', active=False)
            need(self._stationary() and self._handoff_measurements()['passed'], 'ALIGN stop lost original coarse entry')
            retained = check_retained_pose(self.kin.tcp(self.q), self.reference.pose_base[:3, 3],
                self.reference.normal_base, self.reference.pose_base[:3, :3],
                position_tolerance_m=self.executor_config['curve_position_tolerance_m'],
                angle_tolerance_rad=self.executor_config['curve_angle_tolerance_rad'])
            receipt['numerical_curve_entry'] = retained
            need(retained['passed'], 'Unchanged retained curve-entry gate rejected ALIGN')
            receipt.update(alignment_success=True, l3_started=True)
            receipt['alignment_actual_audit'] = self._alignment_audit.report()
            self.control_owner = 'ANALYTIC_L3'; self._alignment_audit = None
            # Original full curve build, force domain, Hall cycle, hold/release.
            FreshExpert._contact_cycle(self)
            receipt['l3_completed'] = True
        except Exception as error:
            receipt['l3_failure_reason'] = str(error)
            raise
        finally:
            audit = getattr(self, '_alignment_audit', None)
            if audit is not None: receipt['alignment_actual_audit'] = audit.report()
            receipt['l3_end_physics_row'] = len(self.env.rows)

    def _tick(self, command, subphase, active=False, recovery=False):
        aligning = self.control_owner == 'ANALYTIC_L3_ALIGN'
        if aligning:
            need(not active and not recovery and not self._contact_active and self.manager.state.phase == Phase.APPROACH,
                 'ALIGN cannot activate contact or recovery')
            history = np.asarray([row['q_command'] for row in self.env.rows[-3:]], float)
            peaks = command_derivatives(history, command)
            need(all(peaks[key] <= LIMITS[key]+1e-8 for key in LIMITS), 'ALIGN whole-stream derivative gate')
            self._alignment_audit.observe(self.kin.tcp(command), 'command')
        start = len(self.env.rows)
        try:
            return super()._tick(command, subphase, active=active, recovery=recovery)
        finally:
            learned = isinstance(self, M2ACTController)
            source = ('given_contact_act_l2_interpolated_v1' if learned else 'given_contact_teacher_l2_v1')
            if self.control_owner == 'ANALYTIC_L3_ALIGN': source = 'given_contact_analytic_l3_align_v1'
            elif self.control_owner == 'ANALYTIC_L3': source = 'given_contact_original_analytic_l3_v1'
            elif self.control_owner not in ('ACT_L2','TEACHER_L2','L2_STOP'): source = 'given_contact_preparation_v1'
            for row in self.env.rows[start:]:
                row.update(task_id=TASK_ID, target_source='given_contact_pose',
                    reference_sha256=self.reference.sha256, reference_revision=0,
                    control_owner=self.control_owner, command_source_id=source,
                    learned_policy=learned, expert_demonstration=not learned)

    def _violations(self, state):
        violations, total = super()._violations(state)
        if self.control_owner == 'ANALYTIC_L3_ALIGN':
            try:
                pose = np.eye(4); pose[:3, :3] = self.env.base[:3, :3].T @ state['rotation_world']
                pose[:3, 3] = state['tcp_base']
                value = self._alignment_audit.observe(pose, 'observed')
                forces = np.asarray([self.env.last_contacts.get(k, 0.) for k in ALL_FORCE_GROUPS], float)
                need(np.isfinite(forces).all() and np.all(forces == 0.), 'ALIGN patient/forbidden physical contact')
                self.env.rows[-1]['given_alignment_observed_audit'] = value
            except ValueError as error:
                violations.append(str(error))
        return violations, total

    def _recover(self):
        return dict(attempted=False, released=False, reason='Given-contact failure stops simulation; no L2/ALIGN fallback')

    def _given_result(self, result):
        result.update(task_id=TASK_ID, target_source='given_contact_pose',
            executed_controller_class=type(self).__module__+'.'+type(self).__name__,
            observation_contract=contract.observation_contract(self.metadata), given_reference=self.estimate_provider.snapshot(),
            l2_handoffs=deepcopy(self.l2_handoffs), l3_initialization='m2_given_contact_l3_align_v1',
            action_semantics_id=ACTION_SEMANTICS_ID,
            raw_target_interpolation_limits=self._trajectory_limits(),
            planning_velocity_numerical_reserve_rad_s=GIVEN_VELOCITY_NUMERICAL_RESERVE_RAD_S,
            planning_jerk_numerical_reserve_rad_s3=(GIVEN_JERK_NUMERICAL_RESERVE_RAD_S3 if self._interior_policy() else 0.),
            raw_target_setters=deepcopy(self._intent_dispatcher.events) if getattr(self,'_intent_dispatcher',None) else [],
            scope='Prescribed full-pose state-only control baseline; no visual, estimator or sensor robustness claim')
        result['ruckig_terminal_normalizations']=deepcopy(getattr(getattr(self,'trajectory',None),'terminal_normalizations',[]))
        if self._interior_policy():result['handoff_policy_id']=handoff.POLICY_ID
        return result

    def _validate_target_reference(self, row):
        need(np.allclose(self.target.position_base, self.reference.pose_base[:3, 3], atol=1e-10, rtol=0) and
             np.allclose(self.rotation_base, self.reference.pose_base[:3, :3], atol=1e-9, rtol=0) and
             np.allclose(self.normal_base, self.reference.normal_base, atol=1e-9, rtol=0),
             'Runtime point/full rotation differs from bound prescribed reference')
        self.reference.verify_sources()


class GivenContactExpert(GivenContactBoundary, FreshExpert):
    def __init__(self, *args, metadata, episode_id, estimate_provider, trajectory_factory=JointTrajectory, **kwargs):
        self.reference = require_route(metadata); self.metadata = deepcopy(metadata)
        self._interior_policy()
        need(isinstance(estimate_provider, GivenPoseProvider), 'Immutable prescribed reference provider required')
        self.episode_id = episode_id; self.estimate_provider = estimate_provider
        self.trajectory_factory = trajectory_factory; self.trajectory = None
        self.control_owner = 'PREPARATION'; self.actor_phase = None; self.prefix = None; self._history = None
        self.l2_handoffs = []; self._l2_started_at = None; self._alignment_audit = None
        super().__init__(*args, **kwargs)
        self.estimate_provider.bind_before_motion(self)

    def _teacher_waypoint(self, destination):
        for _ in range(120*240):
            observation = self._reference_observation()
            if self._interior_policy() and self._handoff_ready(observation):return True
            if len(self.env.rows)%12 == 0:
                if not isinstance(self._history,stage.CausalHistory) or len(self._history.frames) != 2:
                    if self._wait_history_cell(observation) == 'handoff': return True
                    continue
                if (np.max(abs(self.q-destination)) <= 1e-10 and
                        np.max(abs(self.trajectory.velocity)) <= 1e-10 and
                        np.max(abs(self.trajectory.acceleration)) <= 1e-10 and self._stationary()):
                    return False
                self._issue_intent(destination,observation['context'],'given_l2_motion')
            self._advance_tick(self._dispatcher().current['execution_role'])
            observation=self._reference_observation()
            if self._handoff_ready(observation):
                return True
        raise EpisodeRejected('Teacher L2 waypoint timeout')

    def _move_to_prepose(self, row):
        self._point_plan = row
        self._validate_target_reference(row)
        canonical, index = self._canonical(row)
        self.canonical_transfer = canonical.copy(); self.canonical_stage_index = index
        receipt = dict(target=self.target.name, l2_coarse_reached=False, l2_handoff_success=False,
                       expert_l2_fallback=False, l3_started=False, l3_completed=False)
        self.l2_handoffs.append(receipt)
        try:
            need(self.sequence_index == 0 and np.max(abs(self.q)) <= .03 and
                 np.max(abs(self.env.state()['q'])) <= .03, 'Given-contact requires original single-target home')
            self.control_owner = 'PREPARATION'; self._hold(1., 'given_initial_settle')
            need(self._stationary(), 'No actual stationary native history')
            self.trajectory = self.trajectory_factory(self.q, dt=DT, **self._trajectory_limits())
            self.control_owner = 'TEACHER_L2'; self._l2_started_at = float(self.env.state()['time_s'])
            self._history = stage.CausalHistory(self._l2_started_at)
            for destination in canonical[1:]:
                while self._teacher_waypoint(destination):
                    if not self._interior_policy():
                        coarse = self._handoff_measurements(); receipt.update(l2_coarse_reached=True, coarse_entry=coarse)
                    context = self._reference_observation()['context']
                    parked = self._stop_for_handoff(context)
                    if parked is None:continue
                    receipt.update(l2_handoff_success=True, parked_entry=parked, entry=parked)
                    return canonical
            raise EpisodeRejected('Teacher L2 route never reached measured original coarse entry')
        except Exception as error:
            receipt['failure_reason'] = str(error); self.control_owner = 'STOPPED'
            raise

    def run(self, points):
        points = list(points); need(len(points) == 1, 'One prescribed target per episode required')
        return self._given_result(super().run(points))


class GivenContactACT(GivenContactBoundary, M2ACTController):
    def __init__(self, *args, metadata, estimate_provider, **kwargs):
        self.reference = require_route(metadata); self._alignment_audit = None
        need(isinstance(estimate_provider, GivenPoseProvider), 'Immutable prescribed reference provider required')
        super().__init__(*args, metadata=metadata, estimate_provider=estimate_provider, **kwargs)
        self._interior_policy()
        self.estimate_provider.bind_before_motion(self)

    def _advance_tick(self, subphase):
        return GivenContactBoundary._advance_tick(self,subphase)

    def _execute_action(self, context):
        self._deadline(); stamp = float(self.env.state()['time_s']); index = self.prefix.index
        proposal = self.prefix.pop(stamp,context)
        row = dict(time_s=stamp,target=self.target.name,phase=context[-1],prefix_index=index,
            query_index=len(self.l2_queries)-1,proposal=proposal.tolist(),completed=False,
            physics_row_start=len(self.env.rows),action_semantics_id=ACTION_SEMANTICS_ID)
        self.l2_actions.append(row)
        self._issue_intent(proposal,context,'given_l2_motion')
        changed = False
        try:
            for _ in range(12):
                self._advance_tick('given_l2_motion')
                observation = self._observation()
                if observation['context'] != context:
                    self.prefix = None; changed = True; row['interrupted'] = 'context'
                if self._handoff_ready(observation):
                    self.prefix = None; row['interrupted'] = 'handoff'; return 'handoff'
            need(abs(self.env.state()['time_s']-stamp-.05) <= 1e-8, 'Given 20 Hz action clock mismatch')
            row['completed'] = True
            return 'context' if changed else 'continue'
        except Exception as error:
            self.prefix = None; row['failure_reason'] = str(error); raise
        finally:
            row['physics_row_end_exclusive'] = len(self.env.rows)

    def _move_to_prepose(self, row):
        self._validate_target_reference(row)
        self._point_plan = row
        canonical,index = self._canonical(row)
        self.canonical_transfer = canonical.copy(); self.canonical_stage_index = index
        receipt = dict(target=self.target.name,l2_coarse_reached=False,l2_handoff_success=False,
            l3_started=False,l3_completed=False,expert_l2_fallback=False)
        self.l2_handoffs.append(receipt)
        try:
            need(self.sequence_index == 0 and np.max(abs(self.q)) <= .03 and
                 np.max(abs(self.env.state()['q'])) <= .03, 'Given ACT episode must start at original home')
            self.control_owner = 'PREPARATION'; self._hold(1.,'given_initial_settle')
            need(self._stationary(), 'No real stationary native history')
            self.trajectory = self.trajectory_factory(self.q,dt=DT,**self._trajectory_limits())
            self.actor_phase = 'FREE'; self.control_owner = 'ACT_L2'; self.prefix = None
            self._l2_started_at = float(self.env.state()['time_s'])
            self._history = stage.CausalHistory(self._l2_started_at)
            receipt.update(start_time_s=self._l2_started_at,start_physics_row=len(self.env.rows))
            while True:
                self._deadline(); observation = self._observation(); context = observation['context']
                if self._handoff_ready(observation):
                    entry = self._stop_for_handoff(context)
                    if entry is None:continue
                    receipt.update(l2_handoff_success=True,entry=entry,parked_entry=deepcopy(entry),
                        context=list(context),query_count_at_handoff=len(self.l2_queries))
                    return canonical
                if self.prefix is None or not self.prefix.remaining or self.prefix.context != context:
                    self.prefix = None
                    if self._history is None or len(self._history.frames) != 2:
                        self._wait_history_cell(observation)
                        continue
                    self._query(observation)
                self._execute_action(context)
        except Exception as error:
            receipt['failure_reason'] = str(error); self.prefix = None; self.control_owner = 'STOPPED'; raise
        finally:
            receipt['end_physics_row'] = len(self.env.rows)

    def run(self, points):
        return self._given_result(super().run(points))


def make_expert(*args, route, **kwargs):
    """Explicit opt-in native teacher factory; old FreshExpert remains default."""
    need(isinstance(route, dict) and set(route) in ({'schema', 'metadata', 'episode_id'},
         {'schema','metadata','episode_id','handoff_policy_id'}) and
         route['schema'] == TASK_ID, 'Explicit given-contact teacher configuration required')
    need(route.get('handoff_policy_id')==route['metadata'].get('handoff_policy_id') and
         route.get('handoff_policy_id') in (None,handoff.POLICY_ID), 'Runtime/capture handoff policy differs')
    ref = require_route(route['metadata'])
    provider = GivenPoseProvider(ref, route['episode_id'], route['metadata'])
    return GivenContactExpert(*args, metadata=route['metadata'], episode_id=route['episode_id'],
                              estimate_provider=provider, **kwargs)
