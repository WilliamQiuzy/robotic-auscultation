"""CPU tests for physical metrics and evidence boundaries, without model imports."""
import copy
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

    def test_seed_variability_uses_observation_seed_axes_and_zero_increment(self):
        samples = np.zeros((20, 16, 2, 7))
        samples[:, :, 0, 0] = np.arange(16)[None, :]
        reference = np.ones((20, 2, 7))
        reference[:, 0, 1] = 0
        result = d.seed_variability(samples, reference, np.zeros((20, 7)))
        self.assertAlmostEqual(result['per_joint_mean_std_rad'][0], np.std(np.arange(16)))
        self.assertIsNone(result['per_observation_std_to_true_increment'][0][1])
        self.assertEqual(result['zero_true_increment_count'][1], 20)
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
        self.assertAlmostEqual(result['true_delta_from_observation']['full_chunk']['mae_rad'], .0025)

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
            self.assertIn('M0 尚未完成', (Path(td)/'REPORT.md').read_text())
            self.assertIn('Median of per-observation', (Path(td)/'fit_vs_baselines.svg').read_text())
            if 'native_0512' in result['metrics']:
                self.assertAlmostEqual(result['metrics']['native_0512']['absolute']['h0']['mae_rad'],
                                       .00021149022520271826, places=14)


if __name__ == '__main__':
    unittest.main()
