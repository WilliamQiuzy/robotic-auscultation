"""CPU tests for physical metrics and evidence boundaries, without model imports."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.ausc_policy import diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def test_next_command_baseline_starts_at_50ms_not_zero(self):
        q = np.array([[1., -1.]])
        dq = np.array([[2., -4.]])
        np.testing.assert_allclose(d.constant_velocity(q, dq, 3, .05),
                                   [[[1.1, -1.2], [1.2, -1.4], [1.3, -1.6]]])

    def test_chunk_median_and_phase_metrics_do_not_hide_late_error(self):
        q = np.zeros((2, 1))
        ref = np.zeros((2, 3, 1))
        pred = np.array([[[0.], [2.], [4.]], [[0.], [6.], [8.]]])
        metrics = d.chunk_metrics(pred, ref, q, ['FREE', 'HOLD'], .05)
        self.assertEqual(metrics['absolute']['h0']['mae_rad'], 0)
        self.assertAlmostEqual(metrics['absolute']['full_chunk']['mae_rad'], 20/6)
        self.assertEqual(metrics['absolute']['full_chunk']['median_absolute_rad'], 3)
        self.assertEqual(metrics['absolute']['by_phase']['FREE']['full_chunk']['mae_rad'], 2)
        self.assertEqual(metrics['absolute']['per_horizon'][2]['offset_s'], .15)
        self.assertEqual(metrics['delta_from_observation'], metrics['absolute'])
        self.assertNotEqual(metrics['step_delta']['full_chunk']['mae_rad'],
                            metrics['absolute']['full_chunk']['mae_rad'])

    def test_audit_separates_counterfactual_quantiles_and_actual_clip(self):
        raw = np.array([[0., 7.], [1., 7.], [2., 7.]])
        encoded = np.array([[-1.2, 0.], [0., 0.], [1.2, 0.]], np.float32)
        rows = d.dimension_audit(raw, encoded, clamp_applied=False)
        self.assertEqual(rows[0]['actual_clipped_fraction'], 0)
        self.assertAlmostEqual(rows[0]['encoded_outside_unit_fraction'], 2/3)
        self.assertTrue(rows[1]['near_constant'])
        self.assertIsNone(rows[1]['counterfactual_quantile_clipped_fraction'])
        self.assertEqual(sum(rows[0]['bucket_histogram_256']), 3)
        self.assertFalse(rows[0]['bucket_histogram_is_actual'])
        clipped = d.dimension_audit(raw, encoded, clamp_applied=True)
        self.assertAlmostEqual(clipped[0]['actual_clipped_fraction'], 2/3)

    def test_temporal_mapping_rejects_shifted_command_or_discontinuous_frames(self):
        row = dict(sample_index=0, split='train', start=8, label_indices=[8, 9],
                   episode_id='R1', qualification_sha256='a'*64,
                   temporal_label_pairs=[dict(frame_index=20+h, following_frame_index=21+h,
                     physics_interval_inclusive=[239+12*h, 251+12*h],
                     command_physics_row=251+12*h) for h in range(2)])
        out = d.audit_temporal([row], np.array([[1., 1.05]]), .05)
        self.assertEqual(out['command_offsets_s'], [.05, .1])
        self.assertFalse(out['independent_direct_trace_validation'])
        bad = copy.deepcopy(row)
        bad['temporal_label_pairs'][0]['command_physics_row'] -= 12
        with self.assertRaisesRegex(ValueError, 'command'):
            d.audit_temporal([bad], np.array([[1., 1.05]]), .05)
        bad = copy.deepcopy(row)
        bad['temporal_label_pairs'][1]['frame_index'] += 1
        with self.assertRaises(ValueError):
            d.audit_temporal([bad], np.array([[1., 1.05]]), .05)

    def test_seed_variability_uses_observation_seed_axes_and_zero_command_measured_gap(self):
        samples = np.zeros((20, 16, 2, 7))
        samples[:, :, 0, 0] = np.arange(16)[None, :]
        reference = np.ones((20, 2, 7))
        reference[:, 0, 1] = 0
        result = d.seed_variability(samples, reference, np.zeros((20, 7)))
        self.assertAlmostEqual(result['per_joint_mean_std_rad'][0], np.std(np.arange(16)))
        gap = result['command_measured_gap_comparison']
        self.assertIsNone(gap['per_observation_std_to_signal_ratio'][0][1])
        self.assertEqual(gap['per_joint_zero_signal_count'][1], 20)
        self.assertEqual(result['direct_trace_increment']['status'], 'unavailable')
        self.assertFalse(result['command_measured_gap_is_observed_single_step_motion'])
        self.assertNotIn('per_observation_true_increment_rad', result)
        with self.assertRaisesRegex(ValueError, '20 observations.*16 seeds'):
            d.seed_variability(samples[:19], reference[:19], np.zeros((19, 7)))

    def test_prediction_binding_rejects_reordered_reference_even_with_valid_hash(self):
        arrays = {'reference': np.arange(28, dtype=np.float32).reshape(2, 2, 7),
                  'hold': np.zeros((2, 2, 7), np.float32),
                  'state': np.zeros((2, 41), np.float32),
                  'normalized_reference': np.ones((2, 2, 7), np.float32)}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'eval.npz'
            predicted = dict(raw=arrays['reference'].copy(), normalized=np.zeros((2, 2, 7), np.float32),
                reference=arrays['reference'][::-1], hold=arrays['hold'],
                measured=arrays['state'][:, :7], normalized_reference=arrays['normalized_reference'])
            np.savez(p, **predicted)
            meta = dict(array_sha256=d.sha256(p), array_fingerprints={k:d.fingerprint(v) for k,v in predicted.items()},
                        ode_steps=10, prediction_seed_per_example=917)
            p.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'reference.*cache'):
                d.load_predictions(p, arrays)

    def test_nonfinite_metrics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            d.chunk_metrics(np.full((1, 2, 1), np.nan), np.zeros((1, 2, 1)),
                            np.zeros((1, 1)), ['FREE'], .05)

    def test_motion_scale_preserves_small_signal_separate_from_error(self):
        q = np.ones((1, 2))
        reference = np.array([[[1.001, 1.002], [1.003, 1.004]]])
        result = d.chunk_metrics(reference, reference, q, ['FREE'], .05)
        self.assertEqual(result['absolute']['full_chunk']['mae_rad'], 0)
        self.assertAlmostEqual(result['reference_command_measured_gap']['full_chunk']['mae_rad'], .0025)
        self.assertIn('servo tracking error', result['command_measured_gap_definition'])

    def test_servo_gap_and_local_velocity_estimate_are_distinct_signals(self):
        predictions = np.zeros((20, 16, 2, 7))
        predictions[:, :, 0, :] = np.arange(16)[None, :, None]/10000.
        # A persistent .01 rad command/measured gap can exist at zero measured
        # velocity. It cannot establish any command-to-command or measured
        # one-step motion without the original trace.
        result = d.seed_variability(predictions, np.full((20, 2, 7), .01),
                                    np.zeros((20, 7)), measured_dq=np.zeros((20, 7)), period_s=.05)
        gap = result['command_measured_gap_comparison']
        estimate = result['local_velocity_estimate_comparison']
        self.assertEqual(gap['per_observation_signal_rad'][0][0], .01)
        self.assertEqual(estimate['per_observation_signal_rad'][0][0], 0)
        self.assertIsNone(estimate['per_observation_std_to_signal_ratio'][0][0])
        self.assertTrue(estimate['is_estimate'])
        self.assertFalse(estimate['is_observed_trace_increment'])
        self.assertNotEqual(gap['per_joint_std_ge_signal_fraction'], estimate['per_joint_std_ge_signal_fraction'])

    def test_seed_velocity_estimate_uses_radian_per_second_and_rejects_wrong_shape(self):
        predictions = np.zeros((20, 16, 2, 7))
        reference = np.ones((20, 2, 7))
        q = np.zeros((20, 7))
        result = d.seed_variability(predictions, reference, q, measured_dq=np.full((20, 7), -.2), period_s=.05)
        np.testing.assert_allclose(result['local_velocity_estimate_comparison']['per_observation_signal_rad'], .01)
        with self.assertRaisesRegex(ValueError, 'dq shape'):
            d.seed_variability(predictions, reference, q, measured_dq=np.zeros((20, 6)))

    def test_real_cache_export_is_bound_and_does_not_claim_acceptance(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root/'configs/ausc_policy/m0_diagnostics.json'
        config = json.loads(config_path.read_text())
        if not (root/config['prepared_directory']/'cache.npz').exists():
            self.skipTest('Bound local evidence is not distributed with source')
        with tempfile.TemporaryDirectory() as td:
            result = d.run(config_path, root=root, output=td)
            self.assertFalse(result['acceptance']['m0_complete'])
            self.assertFalse(result['temporal_alignment']['independent_direct_trace_validation'])
            self.assertEqual(result['normalization_audit']['actual_native_state_buckets'], 256)
            self.assertTrue(all(r['actual_clipped_fraction'] == 0 for r in result['normalization_audit']['state']))
            self.assertEqual(len(result['normalization_audit']['action_physical']), 7)
            self.assertTrue((Path(td)/'metrics.csv').read_text().startswith('model,representation'))
            with (Path(td)/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            keys = [tuple(row[k] for k in ('model', 'representation', 'phase', 'horizon')) for row in rows]
            self.assertEqual(len(keys), len(set(keys)), 'Each CSV metric must have a unique aggregation key')
            self.assertIn('M0 尚未完成', (Path(td)/'REPORT.md').read_text())
            self.assertIn('Median of per-observation', (Path(td)/'fit_vs_baselines.svg').read_text())
            if 'native_0512' in result['metrics']:
                self.assertAlmostEqual(result['metrics']['native_0512']['absolute']['h0']['mae_rad'],
                                       .00021149022520271826, places=14)

    def test_seed_npz_export_keeps_gap_and_velocity_estimate_explicit(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'configs/ausc_policy/m0_diagnostics.json').read_text())
        cache = root/config['prepared_directory']/'cache.npz'
        if not cache.exists():
            self.skipTest('Bound local evidence is not distributed with source')
        indices = np.arange(20, dtype=np.int64)
        with np.load(cache, allow_pickle=False) as archive:
            reference = archive['reference'][indices]
            measured = archive['state'][indices, :7]
            dq = archive['state'][indices, 7:14]
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            # Deliberately synthetic test fixture, never written to result evidence.
            predictions = np.repeat(reference[:, None], 16, axis=1)
            predictions += np.arange(16, dtype=np.float32)[None, :, None, None]*1e-6
            seed_file = folder/'synthetic_seed_fixture.npz'
            np.savez(seed_file, raw=predictions, reference=reference, measured=measured,
                     sample_indices=indices, seeds=np.arange(917, 933, dtype=np.int64))
            config['seed_predictions'] = str(seed_file)
            cfg_path = folder/'test_config.json'
            cfg_path.write_text(json.dumps(config))
            result = d.run(cfg_path, root=root, output=folder/'output')
            estimate = result['seed_variability']['local_velocity_estimate_comparison']
            np.testing.assert_allclose(estimate['per_observation_signal_rad'], np.abs(dq.astype(float)*.05))
            with (folder/'output'/'seed_variability.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20*7)
            self.assertEqual(rows[0]['observed_trace_increment_available'], 'False')
            text = (folder/'output'/'REPORT.md').read_text()
            self.assertIn('包含伺服跟踪滞后', text)
            self.assertNotIn('真实运动比', text)


if __name__ == '__main__':
    unittest.main()
