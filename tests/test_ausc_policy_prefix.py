"""H10/E5 contract tests without a model, simulator, or trajectory generator."""
import unittest

import numpy as np

from sim.ausc_policy.prefix import ActionPrefix, PrefixContext


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.queue = ActionPrefix()
        self.context = PrefixContext('episode-1', 'R4', 3, 'FREE', 'ACT_L2')
        self.actions = np.arange(70, dtype=np.float64).reshape(10, 7)/1000

    def install(self, **kwargs):
        args = dict(observed_at_s=1., now_s=1., context=self.context)
        args.update(kwargs)
        return self.queue.install(self.actions, **args)

    def pop(self, now, **kwargs):
        args = dict(now_s=now, context=self.context, observation_time_s=now,
                    feedback_time_s=now, hall_time_s=now, target_visible=True, safety_ok=True)
        args.update(kwargs)
        return self.queue.pop(**args)

    def test_first_five_actions_have_original_label_times_and_are_not_repeated(self):
        self.assertEqual(self.install().status, 'installed')
        for k in range(5):
            result = self.pop(1.+k*.05)
            self.assertEqual(result.status, 'action')
            self.assertEqual(result.index, k)
            self.assertAlmostEqual(result.execute_at_s, 1.+(k+1)*.05)
            np.testing.assert_array_equal(result.proposal, self.actions[k])
        self.assertEqual(self.queue.remaining, 0)
        self.assertEqual(self.pop(1.25).reason, 'prefix_exhausted')
        self.assertIsNone(self.pop(1.30).proposal)

    def test_early_or_duplicate_pop_waits_and_does_not_repeat_h0(self):
        self.install()
        self.assertEqual(self.pop(1.).index, 0)
        self.assertEqual(self.pop(1.).status, 'wait')
        self.assertEqual(self.pop(1.01).status, 'wait')
        self.assertEqual(self.pop(1.05).index, 1)
        self.assertEqual(self.queue.remaining, 3)

    def test_original_plan_age_may_exceed_point15_but_feedback_must_be_fresh(self):
        self.install()
        for k in range(5):
            result = self.pop(1.+k*.05)
            self.assertEqual(result.status, 'action')
        self.assertAlmostEqual(result.execute_at_s, 1.25)

    def test_original_observation_cannot_be_relabelled_as_fresh_feedback(self):
        self.install()
        for k in range(4):
            self.pop(1.+k*.05)
        result = self.pop(1.2, observation_time_s=1.)
        self.assertEqual(result.reason, 'stale_observation')
        self.assertEqual(result.status, 'stop')
        self.assertEqual(self.queue.remaining, 0)

    def test_context_events_clear_immediately_and_phase_change_can_requery(self):
        changes = [('episode_id', 'episode-2', 'episode_changed'),
                   ('target', 'L4', 'target_changed'), ('revision', 4, 'revision_changed'),
                   ('control_owner', 'L3', 'control_owner_changed'), ('phase', 'APPROACH', 'phase_changed')]
        for field, value, reason in changes:
            with self.subTest(field=field):
                self.queue = ActionPrefix()
                self.install()
                values = dict(vars(self.context)); values[field] = value
                result = self.queue.sync_context(PrefixContext(**values))
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.status, 'replan')
                self.assertEqual(self.queue.remaining, 0)
                self.assertEqual(self.queue.last_clear_reason, reason)
        approach = PrefixContext('episode-1', 'R4', 3, 'APPROACH', 'ACT_L2')
        result = self.install(context=approach)
        self.assertEqual(result.status, 'installed')
        self.assertEqual(self.pop(1., context=approach).index, 0)

    def test_pop_checks_changed_context_even_before_next_dispatch_time(self):
        self.install(); self.pop(1.)
        changed = PrefixContext('episode-1', 'R4', 4, 'FREE', 'ACT_L2')
        result = self.pop(1.001, context=changed)
        self.assertEqual(result.reason, 'revision_changed')
        self.assertEqual(self.queue.remaining, 0)

    def test_reinstall_unconsumed_prefix_rejects_without_silent_overwrite(self):
        self.install(); self.pop(1.)
        result = self.install()
        self.assertEqual(result.reason, 'prefix_not_consumed')
        self.assertEqual(result.status, 'rejected')
        self.assertEqual(self.queue.remaining, 0)
        self.assertEqual(self.queue.last_clear_reason, 'prefix_not_consumed')

    def test_future_or_nonfinite_stamps_and_out_of_order_clock_clear(self):
        cases = [('observation_time_s', 1.01, 'future_observation'),
                 ('feedback_time_s', 1.01, 'future_feedback'),
                 ('hall_time_s', 1.01, 'future_hall'),
                 ('hall_time_s', np.nan, 'invalid_hall_time'),
                 ('now_s', np.inf, 'invalid_clock')]
        for field, value, reason in cases:
            with self.subTest(field=field):
                self.queue = ActionPrefix(); self.install()
                result = self.pop(1., **{field: value})
                self.assertEqual(result.reason, reason)
                self.assertEqual(self.queue.remaining, 0)
        self.queue = ActionPrefix(); self.install(); self.pop(1.)
        self.assertEqual(self.pop(.999).reason, 'clock_went_backwards')

    def test_original_feedback_and_hall_age_limits_are_not_extended(self):
        for field, age, reason in [('observation_time_s', .150001, 'stale_observation'),
                                  ('feedback_time_s', .050001, 'stale_feedback'),
                                  ('hall_time_s', .050001, 'stale_hall')]:
            with self.subTest(field=field):
                self.queue = ActionPrefix(); self.install()
                result = self.pop(1., **{field: 1.-age})
                self.assertEqual(result.reason, reason)
        self.queue = ActionPrefix(); self.install()
        result = self.pop(1., observation_time_s=.85, feedback_time_s=.95, hall_time_s=.95)
        self.assertEqual(result.status, 'action')

    def test_one_physics_tick_late_is_not_rounded_into_an_on_time_action(self):
        self.install(); self.pop(1.)
        result = self.pop(1.05+1/240)
        self.assertEqual(result.reason, 'missed_command_deadline')
        self.assertIsNone(result.proposal)
        self.assertEqual(self.pop(1.1).reason, 'missed_command_deadline')

    def test_small_float_roundoff_is_accepted(self):
        self.install()
        self.assertEqual(self.pop(1.+1e-9).index, 0)

    def test_simulation_must_remain_at_the_observation_time_during_inference(self):
        self.assertEqual(self.install(now_s=1.01).reason, 'plan_clock_advanced_during_inference')
        self.assertEqual(self.install(now_s=1.2).reason, 'stale_observation')

    def test_invalid_full_chunk_is_not_padded_or_clamped(self):
        for value in [self.actions[:5], np.ones((10, 8)), np.full((10, 7), np.nan), np.full((10, 7), np.inf)]:
            with self.subTest(shape=value.shape):
                result = self.queue.install(value, observed_at_s=1., now_s=1., context=self.context)
                self.assertEqual(result.reason, 'invalid_absolute_chunk')
                self.assertEqual(self.queue.remaining, 0)
        self.assertEqual(self.install().status, 'installed')

    def test_accepted_chunk_is_copied_and_proposals_do_not_reanchor(self):
        original = self.actions.copy()
        self.install()
        self.actions[:] = 99
        first = self.pop(1.)
        np.testing.assert_array_equal(first.proposal, original[0])
        first.proposal[:] = 88
        np.testing.assert_array_equal(self.pop(1.05).proposal, original[1])

    def test_safety_or_visibility_failure_clears_without_fallback(self):
        for args, reason in [({'target_visible': False}, 'target_lost'), ({'safety_ok': False}, 'safety_unsafe')]:
            self.queue = ActionPrefix(); self.install()
            result = self.pop(1., **args)
            self.assertEqual(result.status, 'stop')
            self.assertEqual(result.reason, reason)
            self.assertEqual(self.queue.remaining, 0)

    def test_episode_change_allows_clock_reset_but_not_same_episode_clock_reset(self):
        self.install(); self.pop(1.)
        next_episode = PrefixContext('episode-2', 'R4', 0, 'FREE', 'ACT_L2')
        self.assertEqual(self.queue.sync_context(next_episode).reason, 'episode_changed')
        self.assertEqual(self.install(context=next_episode, observed_at_s=0., now_s=0.).status, 'installed')
        self.assertEqual(self.pop(0., context=next_episode).index, 0)

    def test_cleared_same_observation_cannot_be_reinstalled(self):
        self.install()
        self.queue.clear('explicit_reset')
        self.assertEqual(self.install().reason, 'observation_already_used')

    def test_last_popped_action_keeps_its_execution_interval_before_next_query(self):
        self.install()
        for k in range(5):
            self.pop(1.+k*.05)
        # h4 was dispatched at 1.20 for 1.25. Consuming the queue does not
        # license a second plan for that same physical execution interval.
        result = self.install(observed_at_s=1.2, now_s=1.2)
        self.assertEqual(result.reason, 'previous_action_interval_active')
        self.assertEqual(self.install(observed_at_s=1.25, now_s=1.25).status, 'installed')
        self.assertEqual(self.pop(1.25).execute_at_s, 1.3)

    def test_invalid_context_and_future_install_are_explicit(self):
        invalid = PrefixContext('episode-1', 'R4', True, 'FREE', 'ACT_L2')
        self.assertEqual(self.install(context=invalid).reason, 'invalid_context')
        self.assertEqual(self.install(observed_at_s=1.01).reason, 'future_observation')
        self.assertEqual(self.install(observed_at_s=float('nan')).reason, 'invalid_observation_time')
        contact = PrefixContext('episode-1', 'R4', 3, 'CONTACT', 'L3')
        self.assertEqual(self.install(context=contact).reason, 'phase_not_l2')


if __name__ == '__main__':
    unittest.main()
