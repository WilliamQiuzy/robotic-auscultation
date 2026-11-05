import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from demo import astra_policy, train_act, benchmark


class TrainingContracts(unittest.TestCase):
    def samples(self, n=3):
        states = np.arange(n * 17, dtype=np.float64).reshape(n, 17) / 100
        history = np.stack((states - .01, states), axis=1)
        return dict(states=states, histories=history,
                    actions=np.ones((n, 20, 7)) * .02,
                    origins=states[:, :7].copy(),
                    velocities=(history[:, 1, :7] - history[:, 0, :7]) / .05,
                    observation_times=np.arange(n) * .05)

    def test_disjoint_subject_split_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            train_act.validate_split_ids(['body-a'], ['body-a'])
        train_act.validate_split_ids(['body-a'], ['body-b'])

    def test_history_must_match_current_state(self):
        value = self.samples(); value['histories'][0, 1, 0] += .01
        with self.assertRaises(ValueError):
            train_act.validate_arrays(value)

    def test_nonfinite_actions_rejected(self):
        value = self.samples(); value['actions'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_act.validate_arrays(value)

    def test_normalization_and_pack_match_historical_numerics(self):
        value = self.samples(); train_act.validate_arrays(value)
        stats = train_act.normalization(value)
        batch = train_act.pack(value, stats, np.arange(3))
        h = (value['histories'] - np.array(stats['state_mean'])) / np.array(stats['state_std'])
        np.testing.assert_allclose(batch['observation.state'], h[:, 1, :7], rtol=1e-6)
        expected = np.concatenate([h[:, 1, 7:], h[:, 0]], axis=1)
        np.testing.assert_allclose(batch['observation.environment_state'], expected, rtol=1e-6)
        self.assertEqual(batch['action'].shape, (3, 20, 7))

    def test_fixed_block_origin_is_not_cumulatively_added(self):
        origin = np.arange(7, dtype=float)
        delta = np.ones((20, 7)) * .01
        result = train_act.absolute_chunk(delta, origin)
        np.testing.assert_allclose(result[0], result[-1])
        np.testing.assert_allclose(result[-1], origin + .01)

    def test_manifest_checks_hash_and_complete_window_identity_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            value = self.samples()
            np.savez(root / 'train.npz', **value)
            np.savez(root / 'val.npz', **value)
            manifest = dict(schema='ausc-act-arrays-v1',
                            action_semantics='raw_joint_target_minus_block_origin', splits={})
            for split, filename, subject in [('train', 'train.npz', 'body-a'), ('validation', 'val.npz', 'body-b')]:
                manifest['splits'][split] = dict(file=filename, sha256=train_act.sha256(root / filename),
                                                subject_ids=[subject], window_subject_ids=[subject] * 3)
            p = root / 'arrays.json'; p.write_text(json.dumps(manifest))
            loaded, _ = train_act.load_dataset(p)
            self.assertEqual(len(loaded['train']['states']), 3)
            manifest['splits']['validation']['sha256'] = '0' * 64
            p.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                train_act.load_dataset(p)


class AstraContracts(unittest.TestCase):
    def observation(self):
        return dict(schema='ausc-astra-observation-v1', simulation_only=True,
                    observation_id='frame-001', phase='FREE',
                    q_rad=[0., -.3, 0., .5, 0., 0., 0.],
                    hall_voltage_v=2.51,
                    depth_target_points_m={'R3': [.35, .1, .8]},
                    completed_target_ids=[], pending_target_ids=['R3'],
                    camera={'frame': 'robot_base', 'depth_units': 'm'})

    def decision(self, skill='approach_target'):
        return dict(observation_id='frame-001', skill=skill,
                    target_id='R3', reason='Approach the requested pending target.')

    def test_skill_cannot_invent_anatomical_targets(self):
        d = self.decision(); d['target_id'] = 'R6'
        with self.assertRaises(ValueError):
            astra_policy.validate_decision(d, self.observation())

    def test_late_response_cannot_apply_to_new_observation(self):
        d = self.decision(); d['observation_id'] = 'old-frame'
        with self.assertRaises(ValueError):
            astra_policy.validate_decision(d, self.observation())

    def test_cloud_model_cannot_start_new_approach_during_hold(self):
        obs = self.observation(); obs['phase'] = 'HOLD'
        with self.assertRaises(ValueError):
            astra_policy.validate_decision(self.decision(), obs)
        astra_policy.validate_decision(self.decision('continue_local_control'), obs)

    def test_force_override_is_not_a_supported_field(self):
        d = self.decision(); d['force_N'] = 20
        with self.assertRaises(ValueError):
            astra_policy.validate_decision(d, self.observation())

    def test_real_robot_observation_is_rejected(self):
        obs = self.observation(); obs['simulation_only'] = False
        with self.assertRaises(ValueError):
            astra_policy.build_request(obs)

    def test_payload_has_exact_model_and_strict_function_contract(self):
        value = astra_policy.build_request(self.observation())
        self.assertEqual(value['model'], 'gpt-6-astra')
        self.assertTrue(value['tools'][0]['strict'])
        self.assertFalse(value['parallel_tool_calls'])
        self.assertFalse(value['tools'][0]['parameters']['additionalProperties'])

    def test_mock_transport_proposal_never_claims_execution(self):
        response = SimpleNamespace(output=[SimpleNamespace(type='function_call', name='request_auscultation_skill',
                                                          arguments=json.dumps(self.decision()))],
                                   id='fake-response', usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
        result = astra_policy.propose(client, self.observation())
        self.assertFalse(result['executed'])
        self.assertIsNone(result['physical_success'])

    def test_stopped_controller_cannot_continue_motion(self):
        obs = self.observation(); obs['phase'] = 'FAILED'
        with self.assertRaises(ValueError):
            astra_policy.validate_decision(self.decision('continue_local_control'), obs)


class BenchmarkContracts(unittest.TestCase):
    def test_actual_released_counts_and_identity_scope(self):
        root = Path(__file__).resolve().parents[1] / 'docs' / 'data'
        result = benchmark.build_results(root)
        self.assertEqual(result['expert_qualification']['successes'], 8)
        self.assertEqual(result['expert_qualification']['episodes'], 8)
        self.assertEqual(result['learned_closed_loop']['successes'], 0)
        self.assertEqual(result['learned_closed_loop']['episodes'], 2)
        self.assertEqual(result['independent_heldout']['status'], 'not_reported')
        self.assertIsNone(result['independent_heldout']['success_rate'])
        self.assertFalse(result['act_consumes_rgb_depth_hall'])


if __name__ == '__main__':
    unittest.main()
