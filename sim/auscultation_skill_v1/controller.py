"""Explicit new expert/student action hook over the sealed free-space expert."""
import numpy as np

from sim.curobo_expert_v1.controller import ExpertController as ParentController
from sim.auscultation_expert_layout_v1.controller import DT, CONTROL_STEPS, EpisodeRejected
from sim.auscultation_expert_layout_v1.phase_manager import Phase
from .curve import FixedNormalCurve, force_derived_domain
from .executor import NormalExecutor, command_derivatives, config_values


CLASS = 'sim.auscultation_skill_v1.controller.ExpertController'
CONTACT_DEFAULTS = dict(curve_spacing_m=.00025, curve_position_tolerance_m=2e-5,
                        curve_angle_tolerance_rad=2e-4, contact_clearance_m=.0003,
                        max_inward_braking_m=1e-6, nominal_compression_reserve_m=.0005)


class ExpertController(ParentController):
    def __init__(self, *args, executor_config=None, action_provider=None, condition=None,
                 sensor_condition=None, **kwargs):
        self.executor_config = {**CONTACT_DEFAULTS, **config_values(executor_config)}
        self.action_provider, self.condition = action_provider, condition
        self.sensor_condition = sensor_condition
        if self.sensor_condition is None and condition is not None:
            from .observations import SensorCondition
            self.sensor_condition = SensorCondition(condition)
        self.contact_curves = []
        self.contact_executor = None
        self._contact_active = False
        self._policy_group = 0
        self._previous_desired = 0.
        self._previous_actor_gap = None
        self.command_collision_samples = self.observed_collision_samples = 0
        super().__init__(*args, **kwargs)

    def run(self, points):
        result = super().run(points)
        speed = self.executor_config['normal_speed_limit_m_s']
        inward_speed = self.executor_config['normal_inward_speed_limit_m_s'] * self.speed_scale
        variation = dict(result.get('variation', {}))
        variation.update(free_approach_far_speed_m_s=inward_speed, free_approach_near_speed_m_s=inward_speed,
                         retraction_speed_m_s=speed, fault_response='simulation stops without an alternate motion path',
                         scope='free-space timing retained; contact approach and withdrawal use the shared normal executor')
        variation.pop('fault_retraction_speed_m_s', None)
        actual_policy = dict(action='desired normal velocity in m/s', actor_period_s=CONTROL_STEPS*DT,
                             physics_command_period_s=DT, phase_boundary_refresh=True,
                             normal_speed_limit_m_s=speed, normal_inward_speed_limit_m_s=inward_speed,
                             desired_normal_velocity_bounds_m_s=[-inward_speed, speed],
                             teacher_approach_speed_m_s=-inward_speed,
                             teacher_force_correction_bounds_m_s=[-inward_speed, inward_speed],
                             action_scaling='inward_scaled_outward_fixed_v1',
                             feedback_scaling='Hall correction retains symmetric +/-2mm/s times speed_scale; unload ceiling is fixed',
                             teacher_unload_speed_m_s=speed, expert_force_gain_m_s_per_N=.001,
                             expert_force_deadband_N=.02, force_target_N=4., continuous_hold_s=3.,
                             supervisor='fixed posture/geometry, force/stroke watchdogs, raw joint dynamics, retained curve endpoint stop',
                             actor='expert Hall feedback' if self.action_provider is None else 'external policy; no expert force substitution')
        result.update(executed_controller_class=CLASS, candidate_method=CLASS,
                      candidate_policy='revised_shortcut', executor_config=self.executor_config,
                      contact_curves=self.contact_curves, condition=self.condition,
                      contact_execution_method='fixed normal curve; persistent gap/velocity/acceleration; raw joint finite-difference gates; 240Hz physics and 20Hz actor',
                      expert_force_deadband_N=.02,
                      actual_contact_policy=actual_policy, variation=variation,
                      candidate_scope='simulation; accepted scene/target/pose and free planner retained; new bounded contact executor and explicit actor policy',
                      candidate_timing='free motion retains the parent C2 spatial spline and quintic clock; contact uses persistent scalar dynamics on a checked fixed-frame curve',
                      controller_source='analytic_skill_expert' if self.action_provider is None else 'learned_normal_policy',
                      command_collision_samples=self.command_collision_samples,
                      observed_collision_samples=self.observed_collision_samples,
                      normal_withdrawal_method='same retained curve; bounded normal acceleration to fixed 4mm/s independent of approach speed_scale; exact stationary prepose endpoint',
                      normal_withdrawal_scope='unfiltered measured loaded motion independently audited; no acoustic or clinical inference')
        return result

    def _move_to_prepose(self, row):
        self._point_plan = row
        return super()._move_to_prepose(row)

    def observe_actor(self, state):
        if self.sensor_condition is not None:
            return self.sensor_condition.apply(state, self.target.name)
        return state, dict(observation_time_s=float(state['time_s']), hall_sample_time_s=float(state['time_s']),
                           hall_noise_N=0., observation_age_s=0.)

    def desired_normal_velocity(self, state):
        """Shared action API. A student zero remains zero, including RETRACT."""
        if self.action_provider is not None:
            return float(self.action_provider(self, state))
        maximum = self.executor_config['normal_speed_limit_m_s']
        inward = self.executor_config['normal_inward_speed_limit_m_s'] * self.speed_scale
        phase = self.manager.state.phase.value
        if phase in ('RETRACT', 'DONE'):
            return maximum
        if state['hall_N'] > .3 or state['compression_m'] > .00015:
            error = float(state['hall_N']) - self.config.target_force_N
            return 0. if abs(error) <= .02 else float(np.clip(.001*error, -inward, inward))
        return -inward

    def _skill_observation(self, state, metadata):
        metric = self.metrics(state)
        stamp = metadata['observation_time_s']
        normal_velocity = 0.
        if self._previous_actor_gap is not None:
            old_gap, old_stamp = self._previous_actor_gap
            if stamp > old_stamp:
                normal_velocity = (metric['gap_m']-old_gap)/(stamp-old_stamp)
        self._previous_actor_gap = metric['gap_m'], stamp
        return dict(gap_m=metric['gap_m'], hall_N=float(state['hall_N']),
                    compression_m=float(state['compression_m']), q=np.asarray(state['q']).tolist(),
                    dq=np.asarray(state['dq']).tolist(), normal_base=self.normal_base.tolist(),
                    target_position_base_m=self.target.position_base.tolist(),
                    target_rotation_base=self.rotation_base.tolist(), normal_velocity_m_s=normal_velocity,
                    previous_desired_normal_velocity_m_s=self._previous_desired,
                    executor_gap_m=self.contact_executor.gap,
                    executor_velocity_m_s=self.contact_executor.velocity,
                    executor_acceleration_m_s2=self.contact_executor.acceleration,
                    phase=self.manager.state.phase.value, speed_scale=self.speed_scale)

    def _check_collision(self, q, compression, contact):
        if not np.isfinite(compression):
            raise EpisodeRejected('invalid compression for collision check')
        clearance = self.executor_config['contact_clearance_m'] if contact else .003
        result = self.checker.check(q, clearance=clearance, allow_patient_contact=contact,
                                    compression=max(0., float(compression)))
        distance = float(result.get('minimum_distance_m', float('nan')))
        if result.get('clear') is not True or not np.isfinite(distance) or distance < clearance-1e-9:
            raise EpisodeRejected('command/observation mesh collision rejected: '+str(result))
        return result

    def _tick(self, command, subphase, active=False, recovery=False):
        state = self.env.state()
        if len(self.env.rows) >= 3:
            history = np.asarray([row['q_command'] for row in self.env.rows[-3:]], float)
            peaks = command_derivatives(history, command)
            bounds = dict(velocity=self.joint_speed_limit, acceleration=.5 if self._contact_active else 1., jerk=8.)
            if any(peaks[k] > bounds[k]+1e-8 for k in bounds):
                raise EpisodeRejected('whole-stream joint derivative gate: '+str(peaks))
        self._check_collision(command, state['compression_m'], self._contact_active)
        self.command_collision_samples += 1
        state = super()._tick(command, subphase, active=active, recovery=recovery)
        self._check_collision(state['q'], state['compression_m'], self._contact_active)
        self.observed_collision_samples += 1
        return state

    def _contact_cycle(self):
        self._normal_release_target = None
        self._previous_desired = 0.; self._previous_actor_gap = None
        self._contact_active = True
        loaded_gap = float((self.kin.tcp(np.asarray(self._point_plan['loaded_q']))[:3, 3]-self.target.position_base) @ self.normal_base)
        loaded_compression = float(self._point_plan['loaded_compression_m'])
        domain = force_derived_domain(loaded_gap, loaded_compression, self.normal,
                                      self.config.target_force_N, self.executor_config['nominal_compression_reserve_m'])
        center_gap = domain['nominal_contact_center_gap_m']
        def nominal_check(q, gap):
            compression = max(0., center_gap-gap)
            checked = self.checker.check(q, clearance=self.executor_config['contact_clearance_m'],
                                         allow_patient_contact=True, compression=compression)
            return dict(checked, nominal_compression_m=compression)
        report = dict(target=self.target.name, start_physics_row=len(self.env.rows), motion_complete=False,
                      preflight_passed=False, nominal_contact_center_gap_m=center_gap, nominal_contact_domain=domain)
        self.contact_curves.append(report)
        print('CONTACT_CURVE_START', self.target.name, 'physics_row', len(self.env.rows),
              'minimum_gap_m', domain['minimum_gap_m'], 'maximum_gap_m', self.pre_gap,
              'maximum_nominal_compression_m', domain['maximum_nominal_compression_m'], flush=True)
        try:
            curve = FixedNormalCurve.build(self.kin, self.env.base, self.posture_limits, self.q,
                self.target.position_base, self.normal_base, self.rotation_base, domain['minimum_gap_m'],
                spacing=self.executor_config['curve_spacing_m'],
                position_tolerance=self.executor_config['curve_position_tolerance_m'],
                angle_tolerance=self.executor_config['curve_angle_tolerance_rad'], collision_check=nominal_check)
        except (RuntimeError, ValueError) as error:
            report.update(failure_reason=str(error), preflight_failure=getattr(error, 'evidence', None),
                          end_physics_row_exclusive=len(self.env.rows))
            self._contact_active = False
            raise
        report.update(curve.report, preflight_passed=True)
        print('CONTACT_CURVE_READY', self.target.name, 'physics_row', len(self.env.rows),
              'knots', curve.report['knots'], 'checked_samples', curve.report['verification_samples'], flush=True)
        history = np.asarray([row['q_command'] for row in self.env.rows[-3:]], float)
        self.contact_executor = NormalExecutor(curve, self.executor_config, history, self.speed_scale)
        executor = self.contact_executor
        previous_phase = None; desired = 0.; actor_obs = metadata = None
        phase_ticks = 0; inward_braking = 0.; started_retract = False; final_hold_steps = 0
        try:
            for _ in range(int(np.ceil(80. / self.speed_scale / DT))):
                state = self.env.state(); phase = self.manager.state.phase
                if phase == Phase.ABORT:
                    raise EpisodeRejected('contact state machine aborted: '+str(self.manager.state.failure_reason))
                changed = phase != previous_phase
                if changed:
                    print('CONTACT_PHASE', self.target.name, phase.value, 'physics_row', len(self.env.rows),
                          'gap_m', executor.gap, 'hall_N', float(state['hall_N']), flush=True)
                policy_update = changed or phase_ticks % CONTROL_STEPS == 0
                if policy_update:
                    conditioned, metadata = self.observe_actor(state)
                    if metadata.get('observation_age_s', 0.) > self.config.max_observation_age_s:
                        raise EpisodeRejected('actor observation is stale')
                    actor_obs = self._skill_observation(conditioned, metadata)
                    self.current_skill_observation = actor_obs
                    desired = self.desired_normal_velocity(conditioned)
                    self._previous_desired = desired
                    self._policy_group += 1
                    phase_ticks = 0
                previous_phase = phase
                if phase in (Phase.RETRACT, Phase.DONE):
                    started_retract = True
                released = state['hall_N'] < .2 and state['compression_m'] <= .00015
                remaining = curve.maximum_gap-executor.gap
                if (started_retract and released and desired > 0 and remaining <= .002
                        and remaining > 0 and not executor.stop_pending):
                    executor.plan_stop(curve.maximum_gap)
                before_gap = executor.gap
                if executor.gap == curve.maximum_gap and started_retract and desired >= 0. and executor.velocity == executor.acceleration == 0.:
                    # This is the analytically reached endpoint, not a clipped step.
                    command = executor.hold_stationary()
                    final_hold_steps += 1
                else:
                    command = executor.step(desired)
                    final_hold_steps = 0
                if started_retract:
                    inward_braking += max(0., before_gap-executor.gap)
                    if inward_braking > self.executor_config['max_inward_braking_m'] + 1e-12:
                        raise EpisodeRejected('normal unload inward braking displacement limit')
                subphase = ('normal_release_confirm' if final_hold_steps else
                            'normal_unload' if started_retract and not released else
                            'normal_released_retreat' if started_retract else 'contact_cycle')
                row_index = len(self.control_rows)
                step_succeeded = False
                try:
                    self._block(np.asarray(command)[None], subphase, active=True, gap=executor.gap)
                    step_succeeded = True
                finally:
                    if len(self.control_rows) > row_index:
                        self.control_rows[-1].update(skill_observation=actor_obs, **metadata,
                            policy_update=policy_update, policy_group_id=self._policy_group,
                            desired_normal_velocity_m_s=float(desired), applied_normal_velocity_m_s=executor.velocity,
                            executor_gap_m=executor.gap, executor_velocity_m_s=executor.velocity,
                            executor_acceleration_m_s2=executor.acceleration,
                            executor_stop_mode=executor.mode,
                            endpoint_stop_active=executor.mode == 'endpoint_stop',
                            joint_command_derivative_peaks=executor.peaks,
                            hard_clamp_intervened=False, failure=not step_succeeded,
                            controller_source='analytic_skill_expert' if self.action_provider is None else 'learned_normal_policy')
                phase_ticks += 1
                if final_hold_steps >= int(np.ceil(.2/DT)) and self.manager.state.success:
                    break
            else:
                raise EpisodeRejected('normal contact execution budget exhausted')
            if not self.manager.state.success or self.best_audit_hold < self.config.hold_duration_s-2*DT:
                raise EpisodeRejected('continuous independently verified HOLD/release missing')
            self._normal_release_target = self.target.name
            report.update(motion_complete=True, inward_braking_distance_m=inward_braking)
        finally:
            report['end_physics_row_exclusive'] = len(self.env.rows)
            self._contact_active = False

    def _recover(self):
        # A failed derivative, geometric or observed-state gate does not justify
        # entering the sealed parent's unbounded linear-IK recovery commands.
        return dict(attempted=False, released=False,
                    reason='simulation stopped at failed safety gate; no alternate command path',
                    scope='simulation only; failed episodes excluded from demonstrations')
