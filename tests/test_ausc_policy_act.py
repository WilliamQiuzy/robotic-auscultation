"""Synthetic CPU checks; no model, source dataset, or simulator is loaded."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from sim.ausc_policy import act_baseline as a
except ImportError:
    a = None


class ACTAdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(a, 'M0 ACT adapter has not been implemented')

    def state(self):
        x = np.zeros(41, np.float32)
        x[:7] = np.arange(7) / 10
        x[22] = .9
        x[32:35] = [1, 2, 3]
        x[35:41] = [0, 1, 0, -1, 0, 0]  # Rz(90), column-major
        return x

    def test_small_state_inverts_goal_transform_and_omits_dq_hall(self):
        x = self.state()
        y = a.small_state(x, 'R1', 'FREE')
        self.assertEqual(y.shape, (19,))
        np.testing.assert_allclose(y[7:10], [-2, 1, -3])
        np.testing.assert_allclose(y[10:16], [0, -1, 0, 1, 0, 0])
        x[7:14] = 987
        x[20:22] = 123
        np.testing.assert_array_equal(a.small_state(x, 'R1', 'FREE'), y)
        with self.assertRaisesRegex(ValueError, 'rotation'):
            x[35:41] = 0
            a.small_state(x, 'R1', 'FREE')

    def test_windows_keep_following_command_without_a_second_shift(self):
        n = 14
        arrays = {'times': np.arange(n, dtype=np.float64) * .05,
                  'phases': np.array(['FREE'] * n), 'targets': np.array(['R1'] * n)}
        pairs = [dict(frame_index=i, following_frame_index=i+1,
                      physics_interval_inclusive=[12*i, 12*(i+1)],
                      command_physics_row=12*(i+1)) for i in range(n)]
        mask = np.ones(n, bool)
        self.assertEqual(a.window_starts(arrays, pairs, mask, [], 10), [0, 1, 2, 3, 4])
        self.assertEqual(a.window_starts(arrays, pairs, mask, [9], 10), [])
        mask[9] = False
        self.assertEqual(a.window_starts(arrays, pairs, mask, [], 10), [])
        arrays['phases'][7:] = 'APPROACH'
        self.assertEqual(a.window_starts(arrays, pairs, np.ones(n, bool), [], 10), [])

    def test_nonconsecutive_original_frames_never_form_a_chunk(self):
        n = 10
        arrays = {'times': np.arange(n, dtype=np.float64)*.05,
                  'phases': np.array(['FREE']*n), 'targets': np.array(['R1']*n)}
        pairs = [dict(frame_index=i, following_frame_index=i+1) for i in range(n)]
        pairs[5]['frame_index'] = 100
        self.assertEqual(a.window_starts(arrays, pairs, np.ones(n, bool), [], 10), [])

    def test_stats_do_not_clip_and_keep_zero_span_finite(self):
        state = np.repeat(a.small_state(self.state(), 'R1', 'FREE')[None], 4, axis=0)
        labels = np.ones((4, 10, 7), np.float32)
        stats = a.fit_stats(state, labels)
        outside = labels + 1
        encoded = a.scale(outside, stats['action'])
        self.assertGreater(float(encoded.max()), 1)
        np.testing.assert_allclose(a.unscale(encoded, stats['action']), outside, atol=1e-6)
        self.assertTrue(np.isfinite(a.scale(state, stats['state'])).all())

    def test_pack_preserves_h0_and_inference_contains_no_label(self):
        state = a.small_state(self.state(), 'R1', 'FREE')
        actions = np.arange(70, dtype=np.float32).reshape(10, 7) / 1000
        stats = a.fit_stats(np.repeat(state[None], 2, axis=0), np.stack([actions, actions]))
        rows = [{'state': state, 'actions': actions}]
        batch = a.pack_batch(rows, stats, training=True, rng=np.random.default_rng(0), q_dropout=1)
        np.testing.assert_array_equal(batch['observation.state'], np.zeros((1, 7)))
        np.testing.assert_allclose(a.unscale(batch['action'], stats['action'])[0], actions, atol=1e-6)
        inference = a.pack_batch(rows, stats, training=False)
        self.assertEqual(set(inference), {'observation.state', 'observation.environment_state'})
        self.assertNotIn('action_is_pad', inference)

    def test_npz_members_are_streamed_to_readonly_mmaps_without_rgb(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'episode.npz'
            np.savez_compressed(p, state=np.zeros((3, 41), np.float32), actions=np.ones((3, 7)),
                                times=np.arange(3, dtype=np.float64), phases=np.array(['FREE']*3),
                                targets=np.array(['R1']*3), target_sources=np.array(['rgbd_surface_tracking']*3),
                                rgb=np.ones((3, 100, 100, 3), np.uint8))
            arrays, _ = a.open_episode_mmaps(p, a.sha256(p), Path(td)/'cache')
            self.assertTrue(all(isinstance(x, np.memmap) and not x.flags.writeable for x in arrays.values()))
            self.assertFalse((Path(td)/'cache/rgb.npy').exists())

    def test_metadata_resolves_only_train_paths_and_preserves_exact_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = {'schema': 'wam_vla_training_manifest_v1', 'sealed': True,
                      'horizon': 10, 'action_period_s': .05,
                      'episodes': [dict(episode_id='train_a', split='train', target='R1', path='a.npz',
                                       sha256='a'*64, qualification_path='a.json', qualification_sha256='b'*64,
                                       motion_family='a'),
                                   dict(episode_id='R5_observed_armature02_v1', split='validation', target='R5',
                                        path='/NEVER/OPEN/heldout.npz', motion_family='r5')]}
            sp = root/'source.json'; sp.write_text(json.dumps(source))
            composite = {'schema': 'wam_pi05_v10_recovery_composite_manifest_v1', 'sealed': True,
                         'sources': [dict(id='original10', manifest=dict(path=str(sp), sha256=a.sha256(sp)))],
                         'episodes': [dict(episode_id=e['episode_id'], source_id='original10') for e in source['episodes']]}
            cp = root/'composite.json'; cp.write_text(json.dumps(composite))
            records = a.resolve_train_records(cp, a.sha256(cp), expected_ids=('train_a',))
            self.assertEqual([r['episode_id'] for r in records], ['train_a'])
            self.assertEqual(records[0]['array_path'], str((root/'a.npz').resolve()))
            with self.assertRaisesRegex(ValueError, 'exact TRAIN'):
                a.resolve_train_records(cp, a.sha256(cp), expected_ids=('train_a','missing'))

    def test_l2_requires_raw_align_to_contact_boundary_and_command_match(self):
        arrays = {'state': np.stack([self.state()]), 'actions': np.ones((1, 7), np.float32),
                  'times': np.array([0.], np.float64), 'phases': np.array(['FREE']), 'targets': np.array(['R1'])}
        pairs = [dict(frame_index=0, following_frame_index=1,
                      physics_interval_inclusive=[0, 12], command_physics_row=12)]
        physics = [dict(time_s=i/240, q=arrays['state'][0,:7].tolist(), q_command=[1.]*7,
                        subphase='align', phase='APPROACH', target='R1', recovery=False) for i in range(14)]
        physics[-1]['subphase'] = 'contact_cycle'
        frames = [dict(index=0, physics_row=0, time_s=0., eligible=True, target='R1'),
                  dict(index=1, physics_row=12, time_s=.05, eligible=True, target='R1')]
        valid, receipt = a.l2_mask(arrays, pairs, frames, iter(physics))
        self.assertTrue(valid[0]); self.assertEqual(receipt['first_contact_physics_row'], 13)
        with self.assertRaisesRegex(ValueError, 'align.*contact_cycle'):
            a.l2_mask(arrays, pairs, frames, iter(physics[:-1]))
        broken = copy.deepcopy(physics); broken[12]['q_command'][0] = 2
        with self.assertRaisesRegex(ValueError, 'following command'):
            a.l2_mask(arrays, pairs, frames, iter(broken))

    def test_l2_rejects_an_interior_phase_or_recovery_transition(self):
        state = self.state()
        arrays = {'state': state[None], 'actions': np.ones((1, 7), np.float32),
                  'times': np.array([0.], np.float64), 'phases': np.array(['FREE']), 'targets': np.array(['R1'])}
        pairs = [dict(frame_index=0, following_frame_index=1,
                      physics_interval_inclusive=[0, 12], command_physics_row=12)]
        frames = [dict(index=0, physics_row=0, time_s=0., eligible=True, target='R1'),
                  dict(index=1, physics_row=12, time_s=.05, eligible=True, target='R1')]
        physics = [dict(time_s=i/240, q=state[:7].tolist(), q_command=[1.]*7,
                        subphase='align', phase='APPROACH', target='R1', recovery=False) for i in range(14)]
        physics[-1]['subphase'] = 'contact_cycle'
        physics[4]['phase'] = 'HOLD'
        valid, _ = a.l2_mask(arrays, pairs, frames, iter(physics))
        self.assertFalse(valid[0])
        physics[4]['phase'] = 'APPROACH'; physics[4]['recovery'] = True
        valid, _ = a.l2_mask(arrays, pairs, frames, iter(physics))
        self.assertFalse(valid[0])

    def test_no_model_can_be_constructed_without_owner(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'owner'):
                a.make_native({})

    def test_configuration_is_finite_and_cannot_train_heldout(self):
        config = json.loads(Path('configs/ausc_policy/m0_act.json').read_text())
        a.validate_config(config)
        self.assertEqual(config['modes']['train32']['steps'], 512)
        self.assertEqual(config['modes']['train']['steps'], 2000)
        bad = copy.deepcopy(config); bad['train_episode_ids'][0] = 'R5_observed_armature02_v1'
        with self.assertRaisesRegex(ValueError, 'TRAIN16'):
            a.validate_config(bad)

    def test_error_summary_separates_element_median_from_sample_mae_median(self):
        error = np.zeros((3, 10, 7))
        error[0, -1] = 100
        error[1, -1] = 200
        result = a.error_summary(error)
        self.assertEqual(result['median_absolute_rad'], 0.)
        self.assertEqual(result['median_sample_mae_rad'], 10.)

    def test_block_relative_uses_one_measured_origin_for_all_horizons(self):
        anchors = np.array([[1, 2, 3, 4, 5, 6, 7], [-1, -2, -3, -4, -5, -6, -7]], np.float32)
        delta = np.arange(1, 11, dtype=np.float32)[None, :, None]*.01
        absolute = anchors[:, None, :] + delta
        original = absolute.copy(); anchor_copy = anchors.copy()
        relative = a.encode_actions(absolute, anchors, 'block_relative_joint_radians')
        np.testing.assert_allclose(relative, np.broadcast_to(delta, absolute.shape), atol=3e-7)
        np.testing.assert_allclose(relative[:, -1], .1, atol=3e-7)
        np.testing.assert_array_equal(a.decode_actions(relative, anchors, 'block_relative_joint_radians'), absolute)
        np.testing.assert_array_equal(a.encode_actions(absolute, anchors, 'absolute_joint_radians'), absolute)
        np.testing.assert_array_equal(a.decode_actions(absolute, anchors, 'absolute_joint_radians'), absolute)
        np.testing.assert_array_equal(absolute, original)
        np.testing.assert_array_equal(anchors, anchor_copy)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            a.encode_actions(absolute, np.zeros((2, 10, 7)), 'block_relative_joint_radians')
        with self.assertRaisesRegex(ValueError, 'representation'):
            a.encode_actions(absolute, anchors, 'per_step_delta')

    def test_relative_labels_use_measured_q_despite_state_dropout(self):
        state = a.small_state(self.state(), 'R1', 'FREE')
        anchor = state[:7].copy()
        actions = anchor[None] + np.arange(1, 11, dtype=np.float32)[:, None]*.001
        rows = [dict(state=state, measured_q=anchor, actions=actions)]
        relative = actions-anchor[None]
        stats = a.fit_stats(state[None], relative[None])
        present = a.pack_batch(rows, stats, training=True, action_representation='block_relative_joint_radians')
        hidden = a.pack_batch(rows, stats, training=True, rng=np.random.default_rng(3), q_dropout=1.,
                              action_representation='block_relative_joint_radians')
        np.testing.assert_array_equal(present['action'], hidden['action'])
        np.testing.assert_array_equal(hidden['observation.state'], np.zeros((1, 7)))
        decoded = a.decode_actions(a.unscale(hidden['action'], stats['action']), anchor[None],
                                   'block_relative_joint_radians')
        np.testing.assert_allclose(decoded[0], actions, atol=1e-7)
        np.testing.assert_array_equal(rows[0]['measured_q'], anchor)
        self.assertEqual(hidden['action_is_pad'].shape, (1, 10))
        self.assertFalse(hidden['action_is_pad'].any())

    def test_relative_config_changes_contract_only_and_is_finite(self):
        base = json.loads(Path('configs/ausc_policy/m0_act.json').read_text())
        relative = json.loads(Path('configs/ausc_policy/m0_act_relative_fit32.json').read_text())
        a.validate_config(relative)
        self.assertEqual(relative['mode'], 'train32')
        self.assertEqual(relative['action_representation'], 'block_relative_joint_radians')
        for key in ('seed', 'train_episode_ids', 'horizon', 'execute_steps', 'period_s', 'state_dim',
                    'upstream_commit', 'manifest', 'manifest_sha256', 'model', 'optimizer', 'modes'):
            self.assertEqual(relative[key], base[key])
        for update in ({'mode': 'train'}, {'mode': 'prepare'}, {'prepared_manifest': None}):
            bad = dict(relative, **update)
            with self.assertRaisesRegex(ValueError, 'relative'):
                a.validate_config(bad)

    def test_prepared_manifest_requires_its_own_digest(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'prepared.json'; p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash'):
                a.load_prepared(p, {'prepared_manifest_sha256': '0'*64})

    def test_end_to_end_train16_preparation_never_opens_heldout_or_rgb(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); output = root/'prepared'; output.mkdir()
            records = []
            n = 14
            pairs = [dict(frame_index=i, following_frame_index=i+1,
                          physics_interval_inclusive=[12*i, 12*(i+1)], command_physics_row=12*(i+1))
                     for i in range(n)]
            pair_bytes = (json.dumps(pairs, sort_keys=True, separators=(',', ':'))+'\n').encode()
            import hashlib
            for number, eid in enumerate(a.TRAIN16):
                target = eid.split('_')[0]
                trace = root/eid; (trace/'sensors').mkdir(parents=True)
                q = self.state(); q[:7] += number*.001
                command = np.full(7, number*.01, np.float32)
                arrays = dict(state=np.repeat(q[None], n, axis=0), actions=np.repeat(command[None], n, axis=0),
                              times=np.arange(n, dtype=np.float64)*.05,
                              phases=np.array(['FREE']*n), targets=np.array([target]*n),
                              target_sources=np.array(['rgbd_surface_tracking']*n),
                              rgb=np.ones((n, 2, 2, 3), np.uint8))
                archive = root/(eid+'.npz'); np.savez_compressed(archive, **arrays)
                frames = [dict(index=i, physics_row=i*12, time_s=i*.05, eligible=True, target=target)
                          for i in range(n+1)]
                physics = [dict(time_s=i/240, q=q[:7].tolist(), q_command=command.tolist(),
                                subphase='align' if i <= n*12 else 'contact_cycle',
                                phase='APPROACH', target=target, recovery=False) for i in range(n*12+2)]
                for name, values in [('sensors/frames.jsonl', frames), ('expert_trace.jsonl', physics)]:
                    (trace/name).write_text(''.join(json.dumps(x)+'\n' for x in values))
                qualification = dict(qualified=True, episode_id=eid, arrays_sha256=a.sha256(archive),
                    evidence=dict(label_pairs=pairs, joint_action_derivation=dict(
                        action_contract_version='absolute_joint_command_v1', split='train', period_s=.05,
                        physics_steps=12, no_clipping=True, label_pairs_sha256=hashlib.sha256(pair_bytes).hexdigest(),
                        trace_root=str(trace), trace_sha256={name: a.sha256(trace/name) for name in
                            ('sensors/frames.jsonl','expert_trace.jsonl')},
                        phase_derivation=dict(cross_boundary_label_indices=[]))))
                qp = root/(eid+'.json'); qp.write_text(json.dumps(qualification))
                records.append(dict(episode_id=eid, split='train', target=target, path=archive.name,
                    sha256=a.sha256(archive), qualification_path=qp.name, qualification_sha256=a.sha256(qp),
                    motion_family='family-'+target))
            records += [dict(episode_id=eid, split=split, target=target,
                             path='/NEVER/OPEN/'+target+'.npz', qualification_path='/NEVER/OPEN/report.json',
                             motion_family='family-'+target) for eid, split, target in
                        [('R5_observed_armature02_v1','validation','R5'), ('L5_observed_armature02_v1','test','L5')]]
            source = root/'source.json'
            source.write_text(json.dumps(dict(schema='wam_vla_training_manifest_v1', sealed=True,
                                              horizon=10, action_period_s=.05, episodes=records)))
            composite = root/'manifest.json'
            composite.write_text(json.dumps(dict(schema='wam_pi05_v10_recovery_composite_manifest_v1', sealed=True,
                sources=[dict(id='synthetic', manifest=dict(path=str(source), sha256=a.sha256(source)))],
                episodes=[dict(episode_id=r['episode_id'], source_id='synthetic') for r in records])))
            config = json.loads(Path('configs/ausc_policy/m0_act.json').read_text())
            config.update(manifest=str(composite), manifest_sha256=a.sha256(composite))
            episodes, stats, receipt = a.prepare(config, output)
            self.assertEqual(len(episodes), 16)
            self.assertEqual(receipt['train_windows'], 16*5)
            self.assertFalse(receipt['heldout_arrays_opened'])
            self.assertFalse(list((output/'cache').glob('*/rgb.npy')))
            config['prepared_manifest_sha256'] = a.sha256(output/'prepared.json')
            loaded, _, _ = a.load_prepared(output/'prepared.json', config)
            row = a.row_from_episode(loaded[0], 0, 10)
            np.testing.assert_array_equal(row['actions'], episodes[0]['arrays']['actions'][:10])
            self.assertTrue(all(not x.flags.writeable for e in loaded for x in e['arrays'].values()))
            # Same source caches, state statistics and eligible origins; only action space changes.
            source_bytes = (output/'normalization.json').read_bytes()
            relative_config = dict(config, mode='train32', prepared_manifest=str(output/'prepared.json'),
                                   action_representation='block_relative_joint_radians')
            relative_loaded, relative_stats, source_receipt = a.load_prepared(output/'prepared.json', relative_config)
            self.assertEqual(relative_stats['state'], stats['state'])
            self.assertEqual(source_receipt['action_representation'], 'absolute_joint_radians')
            self.assertNotEqual(relative_stats['action'], stats['action'])
            self.assertEqual((output/'normalization.json').read_bytes(), source_bytes)
            expected = np.stack([e['arrays']['actions'][s:s+10]-e['arrays']['state'][s, :7][None]
                                 for e in relative_loaded for s in e['starts']])
            expected_q01, expected_q99 = np.quantile(expected.reshape(-1, 7), [.01, .99], axis=0)
            np.testing.assert_array_equal(relative_stats['action']['q01'], expected_q01)
            np.testing.assert_array_equal(relative_stats['action']['q99'], expected_q99)
            self.assertFalse(relative_stats['action']['actual_clamp'])
            self.assertTrue(all(not x.flags.writeable for e in relative_loaded for x in e['arrays'].values()))
            relative_output = root/'relative'; relative_output.mkdir()
            a.save_json(relative_output/'normalization.json', relative_stats)
            a.save_action_contract(relative_output, relative_config, source_receipt)
            binding = a.read_json(relative_output/'action_contract.json')
            self.assertEqual(binding['action_representation'], 'block_relative_joint_radians')
            self.assertEqual(binding['prediction_npz_raw'], 'absolute_joint_radians')
            self.assertEqual(binding['normalization']['sha256'], a.sha256(relative_output/'normalization.json'))
            self.assertNotEqual(binding['normalization']['sha256'], source_receipt['normalization']['sha256'])
            self.assertEqual(binding['source_absolute_normalization'], source_receipt['normalization'])
            self.assertEqual(binding['train_windows'], 80)


if __name__ == '__main__':
    unittest.main()
