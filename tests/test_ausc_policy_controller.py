"""D0 orchestration boundaries with CPU fakes; no simulator/model construction."""
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class ControllerTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('sim.ausc_policy.controller')
        except ModuleNotFoundError:
            self.fail('The bounded D0 controller has not been implemented')

    def make(self):
        m = self.module()
        clock = SimpleNamespace(time_s=1., q=np.zeros(7), dq=np.zeros(7), rows=[])

        class Env:
            rows = clock.rows
            base = np.eye(4)
            def state(self):
                return dict(time_s=clock.time_s, q=clock.q.copy(), dq=clock.dq.copy())

        class Kin:
            lower_limits = np.full(7, -2.)
            upper_limits = np.full(7, 2.)
            def tcp(self, q, compression=0.):
                result = np.eye(4)
                result[:3, 3] = np.asarray(q)[:3]+[0, 0, -.025+compression]
                return result

        class Calibration:
            def decode(self, packet, *, now_s, max_age_s=.05):
                if packet.sample_time_s > now_s+1e-9 or now_s-packet.sample_time_s > max_age_s+1e-9:
                    raise ValueError('Hall clock')
                return SimpleNamespace(hall_voltage_v=1., spring_force_estimate_n=2.,
                    displacement_estimate_m=0., sample_time_s=packet.sample_time_s, status='OK')

        class Sensor:
            def __init__(self):
                self.rows = []; self.cal = Calibration(); self.revision = 0; self.distance = .2
                self.hall = SimpleNamespace(read=lambda stamp: self.packet(stamp))
            def packet(self, stamp):
                return m.HallPacket(123, stamp, stamp, 'calibration', 'simulated_adc')
            def capture(self, state, phase, target, physics_row, *, eligible, subphase):
                raw = np.zeros(32); raw[:7] = state['q']; raw[14] = self.distance
                raw[20:23] = [1., 2., 1.]
                self.rows.append(dict(index=len(self.rows), time_s=state['time_s'], target=target,
                    phase=phase, state=raw.tolist(), q=state['q'].tolist(),
                    hall=vars(self.packet(state['time_s'])), body_reference_to_current_base=np.eye(4).tolist(),
                    target_estimate={'revision': self.revision}, eligible=eligible, physics_row=physics_row))

        class Trajectory:
            def __init__(self, q, **limits):
                self.q = np.array(q); self.velocity = np.zeros(7); self.acceleration = np.zeros(7)
                self.targets = []; self.limits = limits
            def set_target(self, q):
                self.target = np.array(q); self.targets.append(self.target.copy())
            def step(self):
                self.q = self.target.copy()
                self.velocity[:] = 0.; self.acceleration[:] = 0.
                return self.q.copy()

        class Client:
            receipt = {'test': 'fake-inference'}
            def __init__(self): self.calls = []; self.advance = 0.
            def predict(self, state, **bindings):
                self.calls.append((state.copy(), bindings))
                clock.time_s += self.advance
                raw = np.repeat((np.arange(10)*.001+.01)[..., None], 7, axis=1)
                return dict(raw=raw.tolist(), latency_ms=1.)

        sensor = Sensor(); client = Client()

        def parent_init(obj, *args, sensor, **kwargs):
            obj.env = Env(); obj.kin = Kin(); obj.sensor = sensor; obj.q = clock.q.copy()
            obj.joint_speed_limit = .175; obj.target = SimpleNamespace(name='R4', position_base=np.zeros(3))
            obj.normal_base = np.array([0., 0., -1.]); obj.rotation_base = np.eye(3)
            obj.task_frames = {'R4': {'task_pose_in_body_reference': np.eye(4).tolist()}}
            obj.control_rows = []; obj.sequence_index = 0; obj._contact_active = False
            obj.contact_curves = []; obj.observed_tcp_speed = 0.
            obj.manager = SimpleNamespace(state=SimpleNamespace(failure_reason=None, phase=m.Phase.APPROACH),
                config=m.PhaseConfig(precontact_gap_m=.025))

        class Harness(m.M0DirectController):
            fail_tick = None
            on_tick = None
            def _block(self, commands, subphase, **kwargs):
                for command in commands:
                    if self.fail_tick == len(clock.rows):
                        raise m.EpisodeRejected('fake physical safety fault')
                    clock.time_s += m.DT; clock.q = np.asarray(command).copy(); self.q = clock.q.copy()
                    clock.rows.append(dict(q_command=self.q.tolist(), time_s=clock.time_s))
                    if self.on_tick is not None: self.on_tick(self)
                    self._capture_observed_tick(self.env.state(), subphase, False, False)
                return self.env.state()
            def _contact_observation(self, state):
                metric, hall, pose = self._tracked_metrics(state)
                observation = SimpleNamespace(normal_gap_m=metric['gap_m'],
                    tangential_error_m=metric['tangent_m'], angle_error_deg=metric['angle_deg'],
                    tcp_speed_m_s=self.observed_tcp_speed)
                return observation, metric, hall

        with patch.object(m.ObservedTeacher, '__init__', parent_init):
            c = Harness(sensor=sensor, client=client, task_id='task', episode_id='episode',
                        trajectory_factory=Trajectory)
        c.actor_phase = 'FREE'; c.control_owner = 'ACT_L2'
        c.trajectory = Trajectory(c.q, velocity=.175, acceleration=.5, jerk=7.99999)
        c._l2_started_at = clock.time_s
        c._capture_l2_frame()
        return m, c, clock, sensor, client

    def test_distinct_first_five_proposals_use_one_trajectory_and_one_query(self):
        m, c, clock, sensor, client = self.make()
        trajectory = c.trajectory
        for _ in range(5): c._l2_action()
        self.assertIs(c.trajectory, trajectory)
        self.assertEqual(len(client.calls), 1)
        np.testing.assert_allclose(np.asarray(trajectory.targets)[:, 0], [.01,.011,.012,.013,.014])
        self.assertEqual(len(clock.rows), 60)
        self.assertEqual([x['prefix_index'] for x in c.l2_actions], list(range(5)))
        np.testing.assert_allclose([x['label_time_s'] for x in c.l2_actions], np.arange(1,6)*.05+1.)
        self.assertTrue(all(frame['eligible'] is False for frame in sensor.rows))

    def enable_hints(self, controller):
        controller.motion_velocity = 'quadratic5'
        controller.trajectory.input = SimpleNamespace(target_velocity=[0.]*7, target_acceleration=[0.]*7)
        controller.trajectory.generator = SimpleNamespace(validate_input=lambda *a, **kw: True)

    def test_hints_leave_positions_and_label_clock_unchanged(self):
        m,c,clock,sensor,client = self.make(); self.enable_hints(c)
        for _ in range(5): c._l2_action()
        np.testing.assert_allclose(np.asarray(c.trajectory.targets)[:,0],[.01,.011,.012,.013,.014])
        np.testing.assert_allclose(c.trajectory.input.target_velocity,.02,atol=1e-12)
        self.assertEqual(len(clock.rows),60)
        self.assertEqual(c.l2_queries[0]['motion_velocity'],'quadratic5')
        self.assertEqual(len(c.l2_actions[0]['target_velocity_rad_s']),7)

    def test_hinted_handoff_explicitly_zeroes_terminal_velocity(self):
        m,c,clock,sensor,client = self.make(); self.enable_hints(c)
        c.trajectory.input.target_velocity=[.1]*7
        c.actor_phase='APPROACH';sensor.distance=.02;c._capture_l2_frame()
        before=c.q.copy();self.assertTrue(c._stop_for_handoff())
        np.testing.assert_array_equal(c.trajectory.input.target_velocity,np.zeros(7))
        np.testing.assert_array_equal(c.q,before)
        self.assertGreaterEqual(len(clock.rows),3)

    def test_invalid_velocity_hint_rejects_before_any_physics_and_does_not_clip(self):
        m,c,clock,sensor,client=self.make(); self.enable_hints(c)
        raw=np.repeat((np.arange(10)*.02)[...,None],7,axis=1)
        client.predict=lambda *a,**kw:dict(raw=raw.tolist())
        with self.assertRaisesRegex(ValueError,'velocity'):
            c._l2_action()
        self.assertEqual(len(clock.rows),0);self.assertEqual(c.prefix.remaining,0)
        np.testing.assert_array_equal(c.l2_queries[0]['raw_absolute_chunk'],raw)

    def test_native_hint_validation_failure_keeps_physics_unadvanced(self):
        m,c,clock,sensor,client=self.make();self.enable_hints(c)
        c.trajectory.generator.validate_input=lambda *a,**kw:False
        with self.assertRaisesRegex(m.EpisodeRejected,'Ruckig'):
            c._l2_action()
        self.assertEqual(len(clock.rows),0)

    def test_hint_context_flush_and_requery_never_reuse_the_old_block(self):
        m,c,clock,sensor,client=self.make();self.enable_hints(c)
        c._l2_action();sensor.revision=1
        sensor.capture(c.env.state(),'FREE','R4',len(clock.rows)-1,eligible=False,subphase='new_revision')
        c._context()
        self.assertIsNone(c._velocity_hints)
        c._l2_action();self.assertEqual(len(client.calls),2)
        self.assertIsNotNone(c._velocity_hints)

    def test_phase_change_discards_unconsumed_prefix_and_requeries(self):
        m, c, clock, sensor, client = self.make()
        c._l2_action()
        sensor.distance = .02
        c._l2_action()
        c._l2_action()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[-1][1]['phase'], 'APPROACH')
        self.assertIn('phase_changed', [event['reason'] for event in c.prefix_events])
        self.assertEqual(c.l2_actions[-1]['prefix_index'], 0)

    def test_revision_change_requeries_without_resetting_trajectory(self):
        m, c, clock, sensor, client = self.make()
        c._l2_action(); sensor.revision = 1
        c._l2_action(); c._l2_action()
        self.assertEqual(client.calls[-1][1]['target_revision'], 1)
        self.assertIn('revision_changed', [event['reason'] for event in c.prefix_events])

    def test_inference_must_not_advance_simulation(self):
        m, c, clock, sensor, client = self.make(); client.advance = m.DT
        with self.assertRaisesRegex(m.EpisodeRejected, 'inference'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 0)
        self.assertEqual(c.prefix.remaining, 0)

    def test_physical_fault_clears_prefix_and_never_retries(self):
        m, c, clock, sensor, client = self.make(); c.fail_tick = 3
        with self.assertRaisesRegex(m.EpisodeRejected, 'physical safety fault'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 3)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(c.prefix.remaining, 0)
        self.assertIn('physical safety fault', c.prefix.last_clear_reason)

    def test_sim_tick_jump_is_rejected(self):
        m, c, clock, sensor, client = self.make()
        c.on_tick = lambda _: setattr(clock, 'time_s', clock.time_s+m.DT)
        with self.assertRaisesRegex(m.EpisodeRejected, 'physics clock'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 1)

    def test_stop_requires_real_three_stationary_rows_and_preserves_retained_q(self):
        m, c, clock, sensor, client = self.make()
        c.actor_phase = 'APPROACH'; sensor.distance = .02
        c._capture_l2_frame()
        before = c.q.copy()
        self.assertTrue(c._stop_for_handoff())
        self.assertGreaterEqual(len(clock.rows), 3)
        np.testing.assert_array_equal(np.asarray([r['q_command'] for r in clock.rows[-3:]]), np.repeat(before[None],3,axis=0))
        np.testing.assert_array_equal(c.q, before)
        self.assertEqual(len(client.calls), 0)

    def test_candidate_gate_is_observed_geometry_not_curve_success(self):
        m, c, clock, sensor, client = self.make()
        c.actor_phase = 'APPROACH'; sensor.distance = .02
        self.assertTrue(c._handoff_candidate())
        clock.q[0] = .004
        self.assertFalse(c._handoff_candidate())
        clock.q[0] = 0.; c.observed_tcp_speed = .02
        self.assertFalse(c._handoff_candidate())

    def test_measured_state19_binding_has_no_anchor_added_again(self):
        m, c, clock, sensor, client = self.make()
        clock.q[:] = .1; c.q = clock.q.copy()
        sensor.rows.clear(); c._capture_l2_frame(); c._l2_action()
        np.testing.assert_array_equal(client.calls[0][0][:7], np.full(7,.1,dtype=np.float32))
        np.testing.assert_allclose(c.trajectory.targets[0], .01)
        self.assertEqual(client.calls[0][1]['episode_id'], 'episode')
        self.assertEqual(client.calls[0][0].shape, (19,))

    def test_expired_l2_does_not_issue_another_command(self):
        m, c, clock, sensor, client = self.make(); clock.time_s = 121.
        with self.assertRaisesRegex(m.EpisodeRejected, '120'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 0)

    def test_revision_event_interrupts_inside_action_and_requeries_current_time(self):
        m, c, clock, sensor, client = self.make()
        def event(obj):
            if len(clock.rows) == 4:
                sensor.revision += 1
                sensor.capture(obj.env.state(), 'FREE', 'R4', 3, eligible=False, subphase='test_event')
        c.on_tick = event
        c._l2_action()
        self.assertEqual(len(clock.rows), 4)
        self.assertEqual(c.l2_actions[-1]['interrupted_reason'], 'revision_changed')
        stamp = clock.time_s
        c._l2_action()
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[-1][1]['sim_stamp_s'], stamp)
        self.assertEqual(c.l2_actions[-1]['prefix_index'], 0)

    def test_future_observation_and_nonfinite_proposal_fail_before_execution(self):
        m, c, clock, sensor, client = self.make()
        sensor.rows[-1]['time_s'] += m.DT
        with self.assertRaisesRegex(m.EpisodeRejected, 'Future'):
            c._l2_action()
        self.assertEqual(len(client.calls), 0)
        sensor.rows.clear(); c._capture_l2_frame()
        client.predict = lambda *a, **kw: {'raw': np.full((10,7),np.nan)}
        with self.assertRaisesRegex(m.EpisodeRejected, 'invalid_absolute_chunk'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 0)

    def test_direct_entry_never_executes_canonical_transfer(self):
        m, c, clock, sensor, client = self.make()
        sensor.distance = .02
        canonical = np.asarray([np.zeros(7),np.ones(7),np.full(7,.5)])
        c._canonical = lambda row: (canonical, 1)
        c._hold = lambda seconds, phase: c._block(np.repeat(c.q[None],3,axis=0), phase)
        c._joint_segment = lambda *args: self.fail('Unexpected canonical/IK transfer')
        returned = c._move_to_prepose({})
        np.testing.assert_array_equal(returned, canonical)
        self.assertTrue(c.d0_handoffs[-1]['stationary_entry'])
        self.assertFalse(c.d0_handoffs[-1]['l3_completed'])
        self.assertTrue(all(np.array_equal(row['q_command'], np.zeros(7)) for row in clock.rows))
        self.assertEqual(c.control_owner, 'MEASURED_ANALYTIC_L3')

    def test_l3_failure_keeps_exact_retained_state_and_d0_failure(self):
        m, c, clock, sensor, client = self.make()
        clock.q[0] = .0001; c.q = clock.q.copy()
        clock.rows.extend(dict(q_command=c.q.tolist(), time_s=clock.time_s) for _ in range(3))
        c.d0_handoffs = [dict(stationary_entry=True, l3_completed=False,
            entry=c._handoff_measurements())]
        before = c.q.copy()
        with patch.object(m.ObservedTeacher, '_contact_cycle', side_effect=RuntimeError('normal curve FK interpolation error')) as call:
            with self.assertRaisesRegex(RuntimeError, 'normal curve FK interpolation error'):
                c._contact_cycle()
        call.assert_called_once()
        np.testing.assert_array_equal(c.q, before)
        self.assertFalse(c.d0_handoffs[-1]['l3_completed'])
        self.assertIn('normal curve FK interpolation error', c.d0_handoffs[-1]['failure_reason'])

    def test_handoff_changed_after_stop_cannot_enter_l3(self):
        m, c, clock, sensor, client = self.make()
        clock.rows.extend(dict(q_command=c.q.tolist(), time_s=clock.time_s) for _ in range(3))
        c.d0_handoffs = [dict(stationary_entry=True, l3_completed=False,
            entry=c._handoff_measurements())]
        clock.time_s += m.DT
        with patch.object(m.ObservedTeacher, '_contact_cycle') as call:
            with self.assertRaisesRegex(m.EpisodeRejected, 'handoff.*changed'):
                c._contact_cycle()
        call.assert_not_called()

    def test_successful_parent_result_is_diagnostic_not_expert_training_data(self):
        m, c, clock, sensor, client = self.make()
        parent = dict(success=True, training_accepted=True, expert_demonstration=True,
            candidate_policy='expert', points=[dict(success=True, training_accepted=True)])
        with patch.object(m.ObservedTeacher, 'run', return_value=parent):
            result = c.run([(c.target,{})])
        self.assertTrue(result['success'])
        for key in ('expert_demonstration','training_dataset_eligible','training_accepted','m0_passed'):
            self.assertIs(result[key], False)
        self.assertIs(result['points'][0]['training_accepted'], False)
        self.assertEqual(result['controller_source'], 'state_ACT_L2 + measured_analytic_L3')
        self.assertNotIn('candidate_policy', result)

    def test_successful_parent_cannot_mark_physics_or_control_traces_as_training(self):
        m, c, clock, sensor, client = self.make()
        c.control_rows.extend([dict(recovery=False, q_command=[.1]*7),
                               dict(recovery=True, q_command=[.2]*7)])
        clock.rows.extend([dict(q_command=[.3]*7, physical_force_N=1.),
                           dict(recovery=True, q_command=[.4]*7, physical_force_N=2.)])
        def parent_run(points):
            for row in c.control_rows+clock.rows:
                row['training_accepted'] = not row.get('recovery', False)
            return dict(success=True, training_accepted=True, points=[dict(success=True)])
        with patch.object(m.ObservedTeacher, 'run', side_effect=parent_run):
            result = c.run([(c.target,{})])
        self.assertTrue(result['success'])
        for row in c.control_rows+clock.rows:
            for key in ('training_accepted', 'training_dataset_eligible', 'expert_demonstration'):
                self.assertIs(row.get(key), False)
        self.assertEqual(c.control_rows[0]['q_command'], [.1]*7)
        self.assertEqual(clock.rows[1]['physical_force_N'], 2.)

    def test_nonfinite_actual_tick_clock_is_rejected(self):
        m, c, clock, sensor, client = self.make()
        c.on_tick = lambda _: setattr(clock, 'time_s', float('nan'))
        with self.assertRaisesRegex(m.EpisodeRejected, 'physics clock'):
            c._l2_action()
        self.assertEqual(len(clock.rows), 1)
        json.dumps(c.prefix_events, allow_nan=False)

    def test_long_episode_identity_does_not_overflow_request_identity(self):
        m, c, clock, sensor, client = self.make()
        c.episode_id = 'x'*128
        c._l2_action()
        self.assertLessEqual(len(client.calls[-1][1]['request_id']), 128)


if __name__ == '__main__':
    unittest.main()
