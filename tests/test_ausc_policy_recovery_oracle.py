import copy
import unittest
import numpy as np

from sim.ausc_policy.recovery_oracle import StationaryRecoveryOracle


class Checker:
    def __init__(self):
        self.calls = 0
        self.fail_at = None

    def check(self, q, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        clear = self.calls != self.fail_at
        return dict(clear=clear, minimum_distance_m=.004 if clear else .002)


class RecoveryOracleTests(unittest.TestCase):
    def setUp(self):
        self.checker = Checker()
        self.planner_calls = []

        def planner(start, goal, rng, deadline):
            self.planner_calls.append(start.copy())
            return [start, goal], dict(reason='direct_checked')

        self.oracle = StationaryRecoveryOracle(self.checker, planner,
            lower=np.full(7, -1.), upper=np.full(7, 1.))
        self.state = dict(q=[.1]*7, dq=[0.]*7, command_history=[[.1]*7]*3,
            hall_load_N=0., physical_contact_N=0., compression_m=0., time_s=2., physics_row=479)
        self.context = dict(target='R4', phase='APPROACH', revision=0, q_goal=[.12]*7)

    def test_current_measured_origin_and_future_chunk(self):
        self.state['q'][0] = .101
        result = self.oracle.query(self.state, self.context)
        self.assertEqual(result['kind'], 'trajectory')
        np.testing.assert_array_equal(self.planner_calls[0], self.state['q'])
        commands = np.asarray(result['commands_240hz'])
        np.testing.assert_allclose(result['action_chunk_relative_rad'],
            commands[np.arange(11, 120, 12)]-self.state['q'], atol=0, rtol=0)
        np.testing.assert_allclose(commands[-1], self.context['q_goal'], atol=0, rtol=0)
        self.assertFalse(result['training_dataset_eligible'])
        self.assertEqual(result['executed_prefix_steps'], 5)
        self.assertEqual(self.checker.last_kwargs,
            dict(clearance=.003, allow_patient_contact=False, compression=0.))

    def test_repeated_query_same_commands(self):
        a = self.oracle.query(self.state, self.context)
        b = self.oracle.query(self.state, self.context)
        self.assertEqual(a['commands_240hz'], b['commands_240hz'])
        self.assertEqual(a['query_sha256'], b['query_sha256'])

    def test_unsupported_states_abstain_before_planning(self):
        changes = [dict(dq=[.006]*7), dict(hall_load_N=.2), dict(physical_contact_N=.2),
            dict(compression_m=.00016), dict(q=[1.01]*7), dict(q=[float('nan')]*7),
            dict(command_history=[[.1]*7, [.11]*7, [.1]*7])]
        for change in changes:
            with self.subTest(change=change):
                state = dict(self.state, **change)
                result = self.oracle.query(state, self.context)
                self.assertEqual(result['kind'], 'abstain')
                self.assertNotIn('action_chunk_relative_rad', result)
        self.assertEqual(self.planner_calls, [])

    def test_contact_or_wrong_revision_is_unsupported(self):
        for change in [dict(phase='CONTACT'), dict(revision=-1), dict(target='R5')]:
            result = self.oracle.query(self.state, dict(self.context, **change))
            self.assertEqual(result['kind'], 'abstain')

    def test_every_dense_command_checked_before_labels(self):
        self.checker.fail_at = 40
        result = self.oracle.query(self.state, self.context)
        self.assertEqual(result['kind'], 'abstain')
        self.assertIn('clearance', result['reason'])
        self.assertNotIn('action_chunk_relative_rad', result)

    def test_missing_path_and_wrong_endpoint_abstain(self):
        for path in [None, [[.4]*7, [.12]*7], [[.1]*7, [.2]*7]]:
            self.oracle.planner = lambda *args: (path, {})
            result = self.oracle.query(self.state, self.context)
            self.assertEqual(result['kind'], 'abstain')

    def test_inputs_not_mutated(self):
        state, context = copy.deepcopy(self.state), copy.deepcopy(self.context)
        self.oracle.query(self.state, self.context)
        self.assertEqual(state, self.state)
        self.assertEqual(context, self.context)


if __name__ == '__main__':
    unittest.main()
