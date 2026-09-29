"""M0 D0 diagnostic: measured-state ACT L2, unchanged measured analytic L3.

No alignment, geometry projection, expert transfer fallback, model imports or
simulator construction. The caller supplies the owned env/sensor/ACTClient.
All executed commands pass through ObservedTeacher's full _block/_tick chain.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sim.auscultation_expert_layout_v1.controller import DT, CONTROL_STEPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase, PhaseConfig
from sim.auscultation_pi05_v6.collect_demo import ObservedTeacher, teacher_policy_description
from sim.auscultation_pi05_v6.execution import JointTrajectory
from sim.auscultation_pi05_v6 import phase_contract
from sim.auscultation_pi05_v6.task_observation import encode_task_state
from sim.auscultation_vla_v2.hall import HallPacket
from .act_baseline import small_state
from .prefix import ActionPrefix, PrefixContext, TIME_TOLERANCE_S

CLASS = 'sim.ausc_policy.controller.M0DirectController'
L2_TIMEOUT_S = 120.
JOINT_LIMITS = dict(velocity=.175, acceleration=.5, jerk=7.99999)


def _record_time(value):
    """Keep a clock fault serializable; None is never accepted as a control time."""
    return (float(value) if isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value) else None)


class M0DirectController(ObservedTeacher):
    """Single-point D0 only. Runtime construction and resource ownership are external."""
    def __init__(self, *args, client, task_id, episode_id,
                 trajectory_factory=JointTrajectory, motion_velocity='zero', posture_hint_preview=False,
                 posture_hint_selection='first_feasible', **kwargs):
        if not callable(getattr(client, 'predict', None)):
            raise ValueError('An explicit ACTClient is required')
        for name, value in (('task_id', task_id), ('episode_id', episode_id)):
            if not isinstance(value, str) or not 0 < len(value) <= 128 or any(not 32 <= ord(c) < 127 for c in value):
                raise ValueError('Invalid '+name)
        self.client, self.task_id, self.episode_id = client, task_id, episode_id
        if motion_velocity not in {'zero','quadratic5'}:
            raise ValueError('Unapproved reference velocity estimator')
        self.motion_velocity = motion_velocity
        if type(posture_hint_preview) is not bool or (posture_hint_preview and motion_velocity != 'quadratic5'):
            raise ValueError('Posture preview requires the declared quadratic5 estimator')
        self.posture_hint_preview = posture_hint_preview
        if (posture_hint_selection not in ('first_feasible','nearest_reference') or
                (posture_hint_selection=='nearest_reference' and not posture_hint_preview)):
            raise ValueError('Endpoint selection requires the posture preview')
        self.posture_hint_selection=posture_hint_selection
        self._velocity_hints = None
        self.trajectory_factory = trajectory_factory
        self.trajectory = None
        self.prefix = ActionPrefix()
        self.actor_phase = None
        self.control_owner = 'PREPARATION'
        self.l2_queries, self.l2_actions, self.prefix_events, self.d0_handoffs = [], [], [], []
        self._l2_started_at = None
        super().__init__(*args, **kwargs)
        if not math.isclose(self.joint_speed_limit, JOINT_LIMITS['velocity'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('M0 requires the original speed_scale=0.7 / 0.175 rad/s contract')

    def _event(self, reason):
        self.prefix_events.append(dict(time_s=_record_time(self.env.state()['time_s']),
            physics_row=len(self.env.rows), target=self.target.name,
            phase=self.actor_phase, control_owner=self.control_owner, reason=reason))

    def _clear(self, reason):
        self.prefix.clear(reason)
        self._velocity_hints = None
        self._event(reason)

    def _deadline(self):
        stamp = self.env.state()['time_s']
        if not np.isfinite(stamp) or stamp < 0:
            raise EpisodeRejected('Invalid L2 simulation clock')
        if self._l2_started_at is None or stamp < self._l2_started_at-TIME_TOLERANCE_S:
            raise EpisodeRejected('L2 simulation clock went backwards')
        if stamp-self._l2_started_at >= L2_TIMEOUT_S-TIME_TOLERANCE_S:
            raise EpisodeRejected('L2 exceeded 120 s simulation timeout; no fallback')

    def _capture_observed_tick(self, state, subphase, active, recovery):
        # Same physical sensor path; no diagnostic frame is eligible expert data.
        due = (not self.sensor.rows or
               state['time_s']-self.sensor.rows[-1]['time_s'] >= .05-1e-8)
        if not due:
            return
        phase = self.manager.state.phase.value if active else (self.actor_phase or 'FREE')
        if phase == 'DONE':
            phase = 'RETRACT'
        if phase not in {'FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT'}:
            raise EpisodeRejected('Invalid diagnostic sensor phase')
        self.sensor.capture(state, phase, self.target.name, len(self.env.rows)-1,
                            eligible=False, subphase=subphase)

    def _capture_l2_frame(self):
        state = self.env.state(); stamp = float(state['time_s'])
        if self.sensor.rows and self.sensor.rows[-1]['time_s'] > stamp+TIME_TOLERANCE_S:
            raise EpisodeRejected('Future L2 observation')
        if (not self.sensor.rows or self.sensor.rows[-1]['target'] != self.target.name or
                abs(self.sensor.rows[-1]['time_s']-stamp) > 1e-8):
            self.sensor.capture(state, self.actor_phase, self.target.name, len(self.env.rows)-1,
                                eligible=False, subphase='act_l2_observation')
        frame = self.sensor.rows[-1]
        if not np.array_equal(np.asarray(frame['q']), np.asarray(state['q'])):
            raise EpisodeRejected('Acquired L2 joints differ from same-time feedback')
        self._tracked_metrics(state)  # mandatory registration and calibrated Hall
        return frame

    def _context(self):
        state = self.env.state()
        if not self.sensor.rows:
            raise EpisodeRejected('Missing L2 observation')
        frame = self.sensor.rows[-1]
        if frame['target'] != self.target.name:
            self._clear('target_changed')
            raise EpisodeRejected('Observation target differs from assigned target')
        for name, stamp, maximum in (('observation', frame['time_s'], .15),
                                     ('feedback', state['time_s'], .05)):
            reason = ActionPrefix._age_reason(name, stamp, state['time_s'], maximum)
            if reason:
                raise EpisodeRejected(reason)
        self.actor_phase = phase_contract.precontact_phase(self.actor_phase, frame['state'])
        confidence = np.asarray(frame['state'], float)[22]
        if not .8 <= confidence <= 1.:
            raise EpisodeRejected('L2 target lost or below original confidence threshold')
        hall = self._read_observed_hall(state)
        if hall.status in {'SCREENED_LIMIT', 'HARD_LIMIT'}:
            raise EpisodeRejected('Hall force limit during L2')
        context = PrefixContext(self.episode_id, self.target.name,
            frame['target_estimate']['revision'], self.actor_phase, self.control_owner)
        event = self.prefix.sync_context(context)
        if event.status == 'stop':
            raise EpisodeRejected(event.reason)
        if event.status == 'replan':
            self._velocity_hints = None
            self._event(event.reason)
        if self.manager.state.failure_reason:
            raise EpisodeRejected('Phase failure: '+self.manager.state.failure_reason)
        return context, frame, hall

    def _query(self, context, frame):
        stamp = float(self.env.state()['time_s']); start_row = len(self.env.rows)
        record = dict(request_id=f'act-query-{len(self.l2_queries)}', time_s=stamp,
            physics_row=start_row, target=context.target, revision=context.revision,
            phase=context.phase, observation_frame=frame['index'], accepted=False,
            query_reason=self.prefix.last_clear_reason, expert_fallback=False)
        self.l2_queries.append(record)
        started = time.perf_counter()
        try:
            state41 = encode_task_state(frame['state'], measured_q=frame['q'],
                hall_packet=HallPacket(**frame['hall']), observation_time_s=frame['time_s'], now_s=stamp,
                body_reference_to_current_base=frame['body_reference_to_current_base'], body_time_s=frame['time_s'],
                task_pose_in_body_reference=self.task_frames[self.target.name]['task_pose_in_body_reference'],
                kinematics=self.kin, hall_calibration=self.sensor.cal)
            state19 = small_state(state41, context.target, context.phase)
            response = self.client.predict(state19, request_id=record['request_id'], task_id=self.task_id,
                episode_id=self.episode_id, frame_id=frame['index'], target_revision=context.revision,
                sim_stamp_s=stamp, target_id=context.target, phase=context.phase)
            if (len(self.env.rows) != start_row or
                    abs(self.env.state()['time_s']-stamp) > TIME_TOLERANCE_S):
                raise EpisodeRejected('Physics advanced during ACT inference')
            record.update(state19=state19.tolist(), raw_absolute_chunk=response['raw'],
                          motion_velocity=self.motion_velocity)
            # ACTClient validates identities/receipt; service has decoded the
            # block-origin anchor ONCE. Never add current q to response['raw'].
            accepted = self.prefix.install(response['raw'], observed_at_s=frame['time_s'],
                                           now_s=stamp, context=context)
            if accepted.status != 'installed':
                raise EpisodeRejected('ACT prefix rejected: '+accepted.reason)
            if self.motion_velocity == 'quadratic5':
                from .motion_hints import velocity_hints
                hint = velocity_hints(response['raw'], method='quadratic5')
                self._velocity_hints = np.asarray(hint['velocity']).copy()
                record['velocity_hint'] = {key:value.tolist() if isinstance(value,np.ndarray) else value
                                           for key,value in hint.items()}
            record.update(state19=state19.tolist(), raw_absolute_chunk=response['raw'],
                response_latency_ms=response.get('latency_ms'), accepted=True)
        except Exception as error:
            record['failure_reason'] = type(error).__name__+': '+str(error)
            raise
        finally:
            record['wall_inference_s'] = time.perf_counter()-started

    def _advance_tick(self, subphase):
        self._deadline()
        stamp = float(self.env.state()['time_s']); row = len(self.env.rows)
        command = self.trajectory.step()
        self._block(np.asarray(command)[None], subphase, active=False)
        state = self.env.state()
        if (len(self.env.rows) != row+1 or not np.isfinite(state['time_s']) or
                abs(state['time_s']-stamp-DT) > 1e-8):
            raise EpisodeRejected('Nonconsecutive actual physics clock/rows')
        if not np.array_equal(np.asarray(self.env.rows[-1]['q_command']), np.asarray(command)):
            raise EpisodeRejected('Executed command differs from real physics history')
        return self._context()[0]

    def _set_motion_target(self, position, velocity=None):
        """Preserve current q/v/a; hints never change the reference time or position."""
        self.trajectory.set_target(position)
        if self.motion_velocity == 'quadratic5':
            hint = np.zeros(7) if velocity is None else np.asarray(velocity,float)
            if hint.shape != (7,) or not np.isfinite(hint).all() or np.max(np.abs(hint)) > JOINT_LIMITS['velocity']:
                raise EpisodeRejected('Invalid bounded target velocity')
            self.trajectory.input.target_velocity = hint.tolist()
            self.trajectory.input.target_acceleration = [0.]*7
            if not self.trajectory.generator.validate_input(self.trajectory.input,
                    check_current_state_within_limits=False,check_target_state_within_limits=True):
                raise EpisodeRejected('Native Ruckig rejected reference velocity/state')

    def _inspect_motion_preview(self, command):
        # Match the exact URDF bounds in _block as well as the original posture
        # envelope. This CPU lookahead never replaces actual per-tick checks.
        if np.any(command < self.kin.lower_limits) or np.any(command > self.kin.upper_limits):
            return dict(clear=False,reason='exact command URDF bounds')
        return self.posture_limits.evaluate(self.kin,self.env.base,command,observed=False,contact=False)

    def _l2_action(self):
        record = None
        try:
            self._deadline()
            self._capture_l2_frame()
            context, frame, hall = self._context()
            if not self.prefix.remaining:
                self._query(context, frame)
            state = self.env.state()
            proposal = self.prefix.pop(now_s=state['time_s'], context=context,
                observation_time_s=frame['time_s'], feedback_time_s=state['time_s'],
                hall_time_s=hall.sample_time_s, target_visible=True, safety_ok=True)
            if proposal.status != 'action':
                raise EpisodeRejected('L2 prefix cannot dispatch: '+proposal.reason)
            if (np.any(proposal.proposal < self.kin.lower_limits) or
                    np.any(proposal.proposal > self.kin.upper_limits)):
                raise EpisodeRejected('Absolute ACT proposal outside joint bounds')
            record = dict(time_s=float(state['time_s']), label_time_s=proposal.execute_at_s,
                prefix_index=proposal.index, query_index=len(self.l2_queries)-1,
                proposal=proposal.proposal.tolist(), physics_row_start=len(self.env.rows),
                target=context.target, phase=context.phase, completed=False)
            self.l2_actions.append(record)
            if self.motion_velocity == 'quadratic5':
                if self._velocity_hints is None or self._velocity_hints.shape != (5,7):
                    raise EpisodeRejected('Missing query-bound velocity hints')
                velocity = self._velocity_hints[proposal.index]
                record['estimated_target_velocity_rad_s'] = velocity.tolist()
                if self.posture_hint_preview:
                    from .motion_preview import choose_velocity_hint
                    try:
                        velocity,preview = choose_velocity_hint(self.trajectory,proposal.proposal,velocity,
                            self._inspect_motion_preview,trajectory_factory=self.trajectory_factory,
                            selection=self.posture_hint_selection)
                        record['posture_hint_preview'] = preview
                    except Exception as error:
                        if hasattr(error,'receipt'):
                            record['posture_hint_preview'] = error.receipt
                        raise
                record['target_velocity_rad_s'] = velocity.tolist()
            else:
                velocity = None
            self._set_motion_target(proposal.proposal,velocity)
            for _ in range(CONTROL_STEPS):
                current = self._advance_tick('act_l2_'+context.phase.lower())
                if current != context:
                    record['interrupted_reason'] = self.prefix.last_clear_reason
                    break
            else:
                record['completed'] = True
            if record['completed'] and abs(self.env.state()['time_s']-proposal.execute_at_s) > TIME_TOLERANCE_S:
                raise EpisodeRejected('Action label clock mismatch')
        except Exception as error:
            self._clear('l2_fault: '+str(error))
            if record is not None:
                record['failure_reason'] = type(error).__name__+': '+str(error)
            raise
        finally:
            if record is not None:
                record['physics_row_end_exclusive'] = len(self.env.rows)

    def _handoff_candidate(self):
        if self.actor_phase != 'APPROACH':
            return False
        observation, _, _ = self._contact_observation(self.env.state())
        config = self.manager.config
        error = abs(observation.normal_gap_m-config.precontact_gap_m)
        # Exact original APPROACH geometry gate, without mutating PhaseManager.
        return bool((error <= config.precontact_tolerance_m or
                     math.isclose(error, config.precontact_tolerance_m, rel_tol=1e-12)) and
                    observation.tangential_error_m <= config.approach_tangential_tolerance_m and
                    observation.angle_error_deg <= config.angle_tolerance_deg and
                    observation.tcp_speed_m_s < config.approach_speed_limit_m_s)

    def _stationary_history(self):
        if len(self.env.rows) < 3:
            return None
        history = np.asarray([row['q_command'] for row in self.env.rows[-3:]], float)
        if (history.shape != (3, 7) or not np.isfinite(history).all() or
                not np.array_equal(history[-1], self.q) or
                not all(np.array_equal(history[0], row) for row in history[1:])):
            return None
        return history

    def _stop_for_handoff(self):
        self.control_owner = 'L2_STOP'
        context, _, _ = self._context()
        self._clear('handoff_candidate_stop')
        # Clear terminal velocity for a stop; never reset current q/v/a.
        self._set_motion_target(self.q.copy())
        try:
            # No reset of Ruckig input q/v/a, no repeat-generated fake history.
            while True:
                current = self._advance_tick('act_l2_handoff_stop')
                if current != context:
                    return False
                stopped = (np.max(np.abs(self.trajectory.velocity)) <= 1e-10 and
                           np.max(np.abs(self.trajectory.acceleration)) <= 1e-10)
                if stopped and self._stationary_history() is not None:
                    self._capture_l2_frame()
                    latest, _, _ = self._context()
                    return latest == context and self._handoff_candidate()
        except Exception as error:
            self._clear('handoff_stop_fault: '+str(error))
            raise
        finally:
            self.control_owner = 'ACT_L2'

    def _handoff_measurements(self):
        state = self.env.state(); metric, hall, measured = self._tracked_metrics(state)
        frame = self.sensor.rows[-1]
        tracked = np.asarray(frame['body_reference_to_current_base']) @ np.asarray(
            self.task_frames[self.target.name]['task_pose_in_body_reference'])
        command = self.kin.tcp(self.q)
        error = command[:3, 3]-self.target.position_base
        gap = float(error @ self.normal_base)
        return dict(time_s=float(state['time_s']), physics_row=len(self.env.rows),
            target=self.target.name, target_revision=frame['target_estimate']['revision'],
            observation_frame=frame['index'],
            command_q=self.q.tolist(), measured_q=np.asarray(state['q']).tolist(),
            measured_dq=np.asarray(state['dq']).tolist(),
            servo_lag_rad=(self.q-np.asarray(state['q'])).tolist(),
            measured_tcp_speed_m_s=self.observed_tcp_speed,
            command_zero_compression_static_frame=dict(gap_m=gap,
                tangent_m=float(np.linalg.norm(error-gap*self.normal_base)),
                full_rotation_error_rad=float(Rotation.from_matrix(command[:3,:3]@self.rotation_base.T).magnitude())),
            measured_hall_tracked_frame=dict(**metric,
                full_rotation_error_rad=float(Rotation.from_matrix(measured[:3,:3]@tracked[:3,:3].T).magnitude()),
                hall_displacement_m=float(hall.displacement_estimate_m)),
            tracked_minus_static_position_m=(tracked[:3,3]-self.target.position_base).tolist(),
            tracked_vs_static_rotation_rad=float(Rotation.from_matrix(tracked[:3,:3]@self.rotation_base.T).magnitude()),
            ruckig_velocity=np.asarray(self.trajectory.velocity).tolist(),
            ruckig_acceleration=np.asarray(self.trajectory.acceleration).tolist(),
            actual_last_three_commands=None if self._stationary_history() is None else self._stationary_history().tolist())

    def _move_to_prepose(self, row):
        self._point_plan = row
        canonical, index = self._canonical(row)  # read only: used by later return-home audit
        self.canonical_transfer, self.canonical_stage_index = canonical.copy(), index
        receipt = dict(target=self.target.name, mode='D0', candidate=False,
            stationary_entry=False, l3_started=False, l3_completed=False, alignment_used=False,
            expert_l2_fallback=False, failure_reason=None)
        self.d0_handoffs.append(receipt)
        try:
            if self.sequence_index != 0 or np.max(np.abs(self.q)) > .03 or np.max(np.abs(self.env.state()['q'])) > .03:
                raise EpisodeRejected('First D0 diagnostic requires the declared initial home')
            # Preparation is not learned transfer; no canonical waypoint is issued.
            if np.max(np.abs(self.q)) > 1e-9:
                if self.checker.edge(self.q, np.zeros(7))['clear'] is not True:
                    raise EpisodeRejected('Initial home preparation bridge rejected')
                self._joint_segment(np.zeros(7), 'initial_home')
            self._hold(1., 'settle')
            if self._stationary_history() is None:
                raise EpisodeRejected('Preparation lacks actual stationary command history')
            self.trajectory = self.trajectory_factory(self.q, **JOINT_LIMITS)
            self.actor_phase = 'FREE'; self.control_owner = 'ACT_L2'
            self._l2_started_at = float(self.env.state()['time_s'])
            receipt.update(l2_start_time_s=self._l2_started_at, l2_start_physics_row=len(self.env.rows))
            while True:
                self._deadline(); self._capture_l2_frame(); self._context()
                if self._handoff_candidate():
                    receipt['candidate'] = True
                    receipt['candidate_measurement'] = self._handoff_measurements()
                    if self._stop_for_handoff():
                        receipt.update(stationary_entry=True, entry=self._handoff_measurements())
                        self.control_owner = 'L3_ENTRY_CHECK'
                        self._context(); self._clear('D0_stopped_candidate_for_L3_entry_check')
                        return canonical
                    self._event('handoff_candidate_lost_requery')
                self._l2_action()
        except Exception as error:
            receipt['failure_reason'] = type(error).__name__+': '+str(error)
            self._clear('D0_l2_failed: '+str(error))
            self.control_owner = 'STOPPED'
            raise
        finally:
            receipt['l2_end_physics_row'] = len(self.env.rows)
            receipt['l2_end_time_s'] = _record_time(self.env.state()['time_s'])
            self.actor_phase = None

    def _contact_cycle(self):
        if not self.d0_handoffs or not self.d0_handoffs[-1]['stationary_entry']:
            raise EpisodeRejected('D0 L3 has no verified real stationary handoff')
        receipt = self.d0_handoffs[-1]
        receipt['l3_started'] = False
        try:
            entry = receipt['entry']; state = self.env.state(); frame = self.sensor.rows[-1]
            if (self._stationary_history() is None or len(self.env.rows) != entry['physics_row'] or
                    not np.isfinite(state['time_s']) or
                    abs(state['time_s']-entry['time_s']) > TIME_TOLERANCE_S or
                    not np.array_equal(self.q, entry['command_q']) or
                    self.target.name != entry['target'] or frame['target'] != entry['target'] or
                    frame['target_estimate']['revision'] != entry['target_revision'] or
                    frame['index'] != entry['observation_frame']):
                raise EpisodeRejected('D0 handoff state/time/target changed after real stop')
            from .contact_entry import check_retained_pose
            gate = check_retained_pose(self.kin.tcp(self.q), self.target.position_base,
                self.normal_base, self.rotation_base,
                position_tolerance_m=self.executor_config['curve_position_tolerance_m'],
                angle_tolerance_rad=self.executor_config['curve_angle_tolerance_rad'])
            receipt['curve_entry_geometry'] = gate
            if not gate['passed']:
                raise EpisodeRejected('D0 retained curve entry rejected: '+', '.join(gate['failures']))
            # A coarse measured candidate is not a curve-compatible command.
            # This necessary check does not replace the original whole-curve
            # IK/FCL preflight or any force, posture and derivative gate.
            self.control_owner = 'MEASURED_ANALYTIC_L3'
            receipt['l3_started'] = True
            # Unchanged measured analytic actor, release hook, curve and executor.
            super()._contact_cycle()
            receipt['l3_completed'] = True
        except Exception as error:
            receipt['failure_reason'] = type(error).__name__+': '+str(error)
            self._clear('D0_l3_failed: '+str(error))
            raise
        finally:
            receipt['l3_end_physics_row'] = len(self.env.rows)

    def run(self, points):
        points = list(points)
        if len(points) != 1:
            raise ValueError('Initial D0 diagnostic supports one declared nominal point per episode')
        result = super().run(points)
        # The parent marks successful physical/control traces as demonstrations.
        # D0 retains those mechanical results, but never qualifies training data.
        for rows in (self.env.rows, self.control_rows):
            for row in rows:
                row.update(training_accepted=False, training_dataset_eligible=False,
                           expert_demonstration=False)
        for key in ('candidate_policy', 'candidate_timing', 'candidate_geometry', 'route_selection',
                    'candidate_scope', 'candidate_method', 'free_motion_command_bounds'):
            result.pop(key, None)
        result.update(executed_controller_class=CLASS, controller_source='state_ACT_L2 + measured_analytic_L3',
            controller_type='m0_D0_diagnostic', dataset_role='m0_diagnostic_not_training_data',
            expert_demonstration=False, training_dataset_eligible=False, training_accepted=False, m0_passed=False,
            actual_contact_policy=teacher_policy_description(), l3_actor='measured_analytic_L3',
            learned_policy_receipt=copy.deepcopy(getattr(self.client, 'receipt', None)),
            l2_queries=self.l2_queries, l2_actions=self.l2_actions, prefix_events=self.prefix_events,
            d0_handoffs=self.d0_handoffs, learned_command_limits=JOINT_LIMITS.copy(),
            learned_interval_timeout_s=L2_TIMEOUT_S, action='absolute H10/E5 50ms ACT targets; service decodes anchor once',
            motion_execution='same continuous Ruckig for L2 and real stop; unchanged NormalExecutor for L3',
            motion_velocity=self.motion_velocity,
            posture_hint_preview=self.posture_hint_preview,
            posture_hint_selection=self.posture_hint_selection,
            reference_tracking_scope='50ms reference updates; label time is not a guarantee of exact waypoint arrival',
            expert_l2_fallbacks=0, alignment_used=False, simulation_paused_for_inference=True,
            realtime_claim=False, preparation_home='initial preparation and final canonical home are outside learned L2')
        for point in result.get('points', []):
            point.update(training_accepted=False, training_dataset_eligible=False, expert_demonstration=False)
        return result
