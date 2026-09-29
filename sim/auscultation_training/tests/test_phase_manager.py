"""Observable-only phase supervision: timing, interruption, and safe completion."""

import importlib
import importlib.util
import math
import unittest


class PhaseManagerTests(unittest.TestCase):
    def setUp(self):
        module_name = "sim.auscultation_training.phase_manager"
        self.assertIsNotNone(importlib.util.find_spec(module_name), "phase manager API is missing")
        self.api = importlib.import_module(module_name)

    def observation(self, **changes):
        fields = dict(hall_force_N=0.0, compression_m=0.0, normal_gap_m=0.04,
                      tangential_error_m=0.0, angle_error_deg=0.0,
                      tcp_speed_m_s=0.0, tcp_position_m=(0.0, 0.0, 0.04))
        fields.update(changes)
        return self.api.ContactObservation(**fields)

    def contact(self, **changes):
        fields = dict(hall_force_N=4.0, compression_m=0.0025, normal_gap_m=0.0)
        fields.update(changes)
        return self.observation(**fields)

    def hold_manager(self):
        manager = self.api.PhaseManager()
        self.assertEqual(manager.update(self.observation(), 0.1).phase, self.api.Phase.CONTACT)
        for _ in range(3):
            state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.HOLD)
        self.assertEqual(state.hold_elapsed, 0.0)
        return manager

    def test_precontact_arrival_does_not_count_as_confirmed_contact(self):
        manager = self.api.PhaseManager()
        self.assertEqual(manager.state.phase, self.api.Phase.APPROACH)
        state = manager.update(self.observation(normal_gap_m=0.06), 0.1)
        self.assertEqual(state.phase, self.api.Phase.APPROACH)
        state = manager.update(self.observation(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.CONTACT)
        self.assertFalse(state.contact_confirmed)
        for _ in range(5):
            state = manager.update(self.observation(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.CONTACT)
        self.assertEqual(state.hold_elapsed, 0.0)

    def test_contact_must_settle_before_hold_and_hold_is_three_continuous_seconds(self):
        manager = self.api.PhaseManager()
        manager.update(self.observation(), 0.1)
        for _ in range(2):
            state = manager.update(self.contact(), 0.1)
            self.assertEqual(state.phase, self.api.Phase.CONTACT)
        state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.HOLD)
        for _ in range(29):
            state = manager.update(self.contact(), 0.1)
            self.assertEqual(state.phase, self.api.Phase.HOLD)
        state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        self.assertAlmostEqual(state.hold_elapsed, 3.0)
        self.assertFalse(state.completed)
        self.assertFalse(state.success)

    def test_fragmented_hold_intervals_never_accumulate(self):
        manager = self.hold_manager()
        for _ in range(20):
            manager.update(self.contact(), 0.1)
        state = manager.update(self.contact(hall_force_N=3.0), 0.1)
        self.assertEqual(state.phase, self.api.Phase.HOLD)
        self.assertEqual(state.hold_elapsed, 0.0)
        for _ in range(3):
            manager.update(self.contact(), 0.1)
        for _ in range(20):
            state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.HOLD)
        self.assertAlmostEqual(state.hold_elapsed, 2.0)

    def test_success_requires_clearance_low_force_and_continuous_release_confirmation(self):
        manager = self.hold_manager()
        for _ in range(30):
            manager.update(self.contact(), 0.1)
        state = manager.update(self.observation(normal_gap_m=0.039), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        state = manager.update(self.observation(hall_force_N=0.2), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        manager.update(self.observation(), 0.1)
        manager.update(self.observation(normal_gap_m=0.039), 0.1)
        state = manager.update(self.observation(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        state = manager.update(self.observation(), 0.05)
        self.assertEqual(state.phase, self.api.Phase.DONE)
        self.assertTrue(state.completed)
        self.assertTrue(state.success)
        self.assertIsNone(state.failure_reason)

    def test_precontact_requires_alignment_and_low_speed(self):
        for changes in ({"normal_gap_m": 0.0}, {"normal_gap_m": 0.037},
                        {"tangential_error_m": 0.0031}, {"angle_error_deg": 5.1},
                        {"tcp_speed_m_s": 0.02}):
            with self.subTest(changes=changes):
                manager = self.api.PhaseManager()
                state = manager.update(self.observation(**changes), 0.1)
                self.assertEqual(state.phase, self.api.Phase.APPROACH)

    def test_force_geometry_compression_and_speed_all_gate_stable_contact(self):
        for changes in ({"hall_force_N": 0.0}, {"hall_force_N": 3.49},
                        {"hall_force_N": 4.51}, {"compression_m": 0.0},
                        {"normal_gap_m": 0.0021}, {"normal_gap_m": -0.0021},
                        {"tangential_error_m": 0.0021}, {"angle_error_deg": 5.1},
                        {"tcp_speed_m_s": 0.005}):
            with self.subTest(changes=changes):
                manager = self.api.PhaseManager()
                manager.update(self.observation(), 0.1)
                for _ in range(5):
                    state = manager.update(self.contact(**changes), 0.1)
                self.assertEqual(state.phase, self.api.Phase.CONTACT)

    def test_interrupted_contact_settle_starts_again(self):
        manager = self.api.PhaseManager()
        manager.update(self.observation(), 0.1)
        manager.update(self.contact(), 0.1)
        manager.update(self.contact(), 0.1)
        manager.update(self.contact(tcp_speed_m_s=0.02), 0.1)
        for _ in range(2):
            self.assertEqual(manager.update(self.contact(), 0.1).phase, self.api.Phase.CONTACT)
        self.assertEqual(manager.update(self.contact(), 0.1).phase, self.api.Phase.HOLD)

    def test_each_invalid_hold_observable_resets_continuous_hold_and_resettles(self):
        for changes in ({"hall_force_N": 3.49}, {"compression_m": 0.0},
                        {"normal_gap_m": 0.003}, {"tangential_error_m": 0.003},
                        {"angle_error_deg": 6.0}, {"tcp_speed_m_s": 0.005}):
            with self.subTest(changes=changes):
                manager = self.hold_manager()
                manager.update(self.contact(), 0.1)
                state = manager.update(self.contact(**changes), 0.1)
                self.assertEqual(state.phase, self.api.Phase.HOLD)
                self.assertEqual(state.hold_elapsed, 0.0)
                for _ in range(3):
                    self.assertEqual(manager.update(self.contact(), 0.1).hold_elapsed, 0.0)
                self.assertAlmostEqual(manager.update(self.contact(), 0.1).hold_elapsed, 0.1)

    def test_faults_latch_failure_and_retract_before_abort(self):
        cases = [({"hall_force_N": 5.5}, 0.1, "overforce"),
                 ({"compression_m": 0.00725}, 0.1, "overcompression"),
                 ({"observation_age_s": 0.201}, 0.1, "stale_observation"),
                 ({"target_valid": False}, 0.1, "target_lost"),
                 ({}, 0.201, "watchdog_timeout"),
                 ({"angle_error_deg": math.nan}, 0.1, "invalid_observation")]
        for changes, dt, reason in cases:
            with self.subTest(reason=reason):
                manager = self.hold_manager()
                state = manager.update(self.contact(**changes), dt)
                self.assertEqual(state.phase, self.api.Phase.RETRACT)
                self.assertEqual(state.failure_reason, reason)
                self.assertFalse(state.completed)
                self.assertFalse(state.success)
                state = manager.update(self.observation(target_valid=False), 0.1)
                self.assertEqual(state.phase, self.api.Phase.RETRACT)
                state = manager.update(self.observation(), 0.05)
                self.assertEqual(state.phase, self.api.Phase.ABORT)
                self.assertTrue(state.completed)
                self.assertFalse(state.success)
                self.assertEqual(state.failure_reason, reason)

    def test_fault_during_successful_retraction_cannot_become_done(self):
        manager = self.hold_manager()
        for _ in range(30):
            manager.update(self.contact(), 0.1)
        state = manager.update(self.observation(observation_age_s=0.201), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        manager.update(self.observation(), 0.1)
        state = manager.update(self.observation(), 0.05)
        self.assertEqual(state.phase, self.api.Phase.ABORT)
        self.assertFalse(state.success)

    def test_approach_contact_and_hold_timeouts_request_failed_retraction(self):
        for phase in ("approach", "contact", "hold"):
            with self.subTest(phase=phase):
                config = self.api.PhaseConfig(**{phase + "_timeout_s": 0.5})
                manager = self.api.PhaseManager(config)
                if phase != "approach":
                    manager.update(self.observation(), 0.1)
                if phase == "hold":
                    for _ in range(3):
                        manager.update(self.contact(), 0.1)
                for _ in range(5):
                    obs = self.contact(hall_force_N=3.0) if phase == "hold" else self.observation(normal_gap_m=0.06)
                    state = manager.update(obs, 0.1)
                self.assertEqual(state.phase, self.api.Phase.RETRACT)
                self.assertEqual(state.failure_reason, phase + "_timeout")

    def test_retraction_timeout_aborts_even_after_complete_hold(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(retract_timeout_s=0.2))
        manager.update(self.observation(), 0.1)
        for _ in range(33):
            manager.update(self.contact(), 0.1)
        self.assertEqual(manager.state.phase, self.api.Phase.RETRACT)
        manager.update(self.contact(), 0.1)
        state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.ABORT)
        self.assertEqual(state.failure_reason, "retract_timeout")
        self.assertFalse(state.success)

    def test_episode_deadline_is_not_reset_by_phase_transitions(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(episode_timeout_s=0.5))
        manager.update(self.observation(), 0.1)
        for _ in range(3):
            manager.update(self.contact(), 0.1)
        state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        self.assertEqual(state.failure_reason, "episode_timeout")

    def test_episode_deadline_during_retraction_latches_failure_and_still_confirms_release(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(episode_timeout_s=3.5))
        manager.update(self.observation(), 0.1)
        for _ in range(33):
            manager.update(self.contact(), 0.1)
        self.assertEqual(manager.state.phase, self.api.Phase.RETRACT)
        self.assertIsNone(manager.state.failure_reason)
        state = manager.update(self.observation(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        self.assertEqual(state.failure_reason, "episode_timeout")
        self.assertFalse(state.completed)
        state = manager.update(self.observation(), 0.05)
        self.assertEqual(state.phase, self.api.Phase.ABORT)
        self.assertTrue(state.completed)
        self.assertFalse(state.success)
        self.assertEqual(state.failure_reason, "episode_timeout")

    def test_episode_deadline_during_retraction_keeps_retraction_timeout_bounded(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(episode_timeout_s=3.5,
                                                            retract_timeout_s=0.4))
        manager.update(self.observation(), 0.1)
        for _ in range(33):
            manager.update(self.contact(), 0.1)
        for _ in range(3):
            state = manager.update(self.contact(), 0.1)
            self.assertEqual(state.phase, self.api.Phase.RETRACT)
            self.assertEqual(state.failure_reason, "episode_timeout")
        state = manager.update(self.contact(), 0.1)
        self.assertEqual(state.phase, self.api.Phase.ABORT)
        self.assertFalse(state.success)
        self.assertEqual(state.failure_reason, "episode_timeout; retract_timeout")

    def test_invalid_numeric_observations_fail_closed_without_raising(self):
        invalid = [None, "unexpected"]
        for changes in ({"hall_force_N": math.inf}, {"compression_m": -0.001},
                        {"normal_gap_m": math.nan}, {"tangential_error_m": -1.0},
                        {"angle_error_deg": "zero"}, {"tcp_speed_m_s": -1.0},
                        {"observation_age_s": -1.0}, {"target_valid": "true"},
                        {"tcp_position_m": (0, 0)}, {"tcp_position_m": (0, math.nan, 0)}):
            invalid.append(self.observation(**changes))
        for observation in invalid:
            with self.subTest(observation=observation):
                manager = self.api.PhaseManager()
                state = manager.update(observation, 0.1)
                self.assertEqual(state.phase, self.api.Phase.RETRACT)
                self.assertEqual(state.failure_reason, "invalid_observation")

    def test_invalid_dt_is_rejected_without_mutating_state(self):
        manager = self.api.PhaseManager()
        before = manager.state
        for dt in (-0.1, math.nan, math.inf, "0.1", True):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    manager.update(self.observation(), dt)
                self.assertEqual(manager.state, before)

    def test_zero_dt_never_credits_confirmation_time(self):
        manager = self.api.PhaseManager()
        manager.update(self.observation(), 0.0)
        for _ in range(10):
            state = manager.update(self.contact(), 0.0)
        self.assertEqual(state.phase, self.api.Phase.CONTACT)
        self.assertEqual(state.hold_elapsed, 0.0)

    def test_terminal_state_and_returned_snapshots_do_not_change(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(retract_timeout_s=0.1))
        initial = manager.state
        manager.update(self.observation(target_valid=False), 0.1)
        terminal = manager.update(self.contact(), 0.1)
        self.assertEqual(terminal.phase, self.api.Phase.ABORT)
        self.assertEqual(manager.update(self.observation(), 0.1), terminal)
        self.assertEqual(initial.phase, self.api.Phase.APPROACH)

    def test_unsafe_or_nonfinite_configuration_is_rejected(self):
        for fields in ({"target_force_N": math.nan}, {"hold_duration_s": 0.0},
                       {"force_tolerance_N": -0.1}, {"hard_force_limit_N": 4.5},
                       {"max_compression_m": 0.00001}, {"precontact_gap_m": 0.001},
                       {"retract_clearance_m": -0.04}, {"max_observation_age_s": math.inf}):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    self.api.PhaseConfig(**fields)

    def test_inclusive_geometry_boundaries_do_not_fail_from_subtraction_roundoff(self):
        for gap in (0.038, 0.042):
            with self.subTest(gap=gap):
                manager = self.api.PhaseManager()
                state = manager.update(self.observation(normal_gap_m=gap, tangential_error_m=0.003,
                                                        angle_error_deg=5.0), 0.1)
                self.assertEqual(state.phase, self.api.Phase.CONTACT)

    def test_unrepresentable_numeric_inputs_fail_closed(self):
        too_large = 10 ** 1000
        manager = self.api.PhaseManager()
        with self.assertRaises(ValueError):
            manager.update(self.observation(), too_large)
        state = manager.update(self.observation(hall_force_N=too_large), 0.1)
        self.assertEqual(state.phase, self.api.Phase.RETRACT)
        self.assertEqual(state.failure_reason, "invalid_observation")

    def test_stale_or_missing_release_observations_cannot_confirm_retraction(self):
        manager = self.api.PhaseManager(self.api.PhaseConfig(retract_timeout_s=0.4))
        manager.update(self.observation(target_valid=False), 0.1)
        manager.update(self.observation(), 0.1)
        manager.update(None, 0.1)
        self.assertEqual(manager.update(self.observation(), 0.1).phase, self.api.Phase.RETRACT)
        state = manager.update(self.observation(observation_age_s=0.201), 0.1)
        self.assertEqual(state.phase, self.api.Phase.ABORT)
        self.assertEqual(state.failure_reason, "target_lost; retract_timeout")
        self.assertFalse(state.success)

    def test_one_sample_cannot_enter_contact_and_hold_together(self):
        config = self.api.PhaseConfig(contact_normal_tolerance_m=0.05, settle_duration_s=0.01)
        manager = self.api.PhaseManager(config)
        state = manager.update(self.contact(normal_gap_m=0.04), 0.1)
        self.assertEqual(state.phase, self.api.Phase.CONTACT)
        self.assertEqual(state.hold_elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
