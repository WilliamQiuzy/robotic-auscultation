"""Closed ACT9 research claims and actual learned media, without private runtime data."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import unittest


DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
LEDGER = DRAFT / 'evidence/act-recovery-iteration9-closed-20261107.public.json'


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class ACT9ClosedPresentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()

    def evidence(self):
        self.assertTrue(LEDGER.is_file(), 'Closed ACT9 outcomes need their own public ledger.')
        return json.loads(LEDGER.read_text())

    def test_all_four_normal_cases_keep_the_selection_failures(self):
        d = self.evidence()['known_development']
        expected = [('p001120', 'TRAIN', 'R3', True), ('p001120', 'TRAIN', 'L3', True),
                    ('p000923', 'selection_VAL', 'R3', False), ('p000923', 'selection_VAL', 'L3', False)]
        self.assertEqual([(c['body'], c['split'], c['target'], c['physical_pass']) for c in d['cases']], expected)
        self.assertEqual((d['planned'], d['executed'], d['independently_physics_qualified']), (4, 4, 2))
        self.assertEqual(sum(c['physical_pass'] for c in d['cases']), d['independently_physics_qualified'])
        self.assertTrue(d['owner_closed_normally'] and d['all_workers_closed_normally'] and d['source34_stable'])
        self.assertEqual(d['infrastructure_errors'], 0)
        for c in d['cases']:
            if c['physical_pass']:
                self.assertTrue(c['release_and_home_passed'])
            else:
                self.assertEqual(c['failure'], 'original_L2_120_second_deadline')
        self.assertEqual(self.evidence()['fresh_benchmark'], {'identities': 8, 'declared_scenarios': 32, 'executed_cases': 0, 'payloads_opened': False})
        self.assertFalse(self.evidence()['model_accepted'])

    def test_training_protocol_and_valid_prefix_fit_do_not_claim_generalization(self):
        e = self.evidence()
        t = e['training']
        self.assertEqual(t['initialization'], {'treatment': 'ACT6', 'selected_step': 18000, 'fresh_AdamW': True, 'initial_optimizer_step': 0})
        self.assertEqual((t['new_updates'], t['selected_new_update']), (1000, 1000))
        self.assertTrue(t['own_full_CPU_equivalence_passed'])
        self.assertFalse(t['new_updates_are_lifetime_total'])
        self.assertEqual(t['selection_rule'], 'fixed_final_step_not_validation_minimum')
        m = e['model_protocol']
        self.assertEqual((m['state_dimension'], m['observation_history'], m['action_horizon'], m['executed_cells']), (17, 2, 20, 10))
        self.assertEqual(m['posture_head'], 'smooth_posture_head_v2')
        self.assertEqual(m['learned_motor_stage'], 'L2_approach')
        self.assertEqual(m['contact_controller'], 'analytic_Hall_L3')
        self.assertFalse(m['expert_L2_fallback'] or m['RGBD_neural_inputs'] or m['Hall_neural_input'] or m['audio'])
        self.assertEqual(e['selected_model'], {'bytes': 11278593, 'sha256': '7523475be10b1ad9edb1b2080513c748d7d3d0dcbc90b19fbefc5539ac27baec'})
        f = e['measured_TRAIN_fit']
        self.assertEqual((f['windows'], f['valid_future_cells'], f['horizon20_labels']), (58, 479, 0))
        self.assertEqual(f['scope'], 'valid_TRAIN_prefixes_only_not_full_H20_or_heldout_fit')
        self.assertAlmostEqual(f['ACT_MAE_mrad'], 2.0598296290110505)
        self.assertEqual((f['HOLD_MAE_mrad'], f['constant_velocity_MAE_mrad']), (21.23748, 13.66792))
        for key in ['population_generalization', 'overall_improvement_over_ACT6', 'algorithm_causal_effect', 'hardware_validation']:
            self.assertFalse(e['limits'][key])

    def test_unchanged_gates_and_actual_L3_scalars(self):
        e = self.evidence()
        p = e['unchanged_physics']
        self.assertEqual(p['normal_hold_N'], [3.5, 4.5])
        self.assertEqual((p['continuous_hold_min_s'], p['total_contact_max_N'], p['Hall_strict_max_N'], p['stroke_max_mm']), (3, 6, 5.5, 6))
        self.assertEqual((p['C_protrusion_mm'], p['B_recess_mm'], p['TCP_mm'], p['camera_down_degrees']), (5, 1, 132, 7))
        self.assertEqual((p['forbidden_force_exact_N'], p['disk_plane_tilt_max_deg'], p['body_condition']), (0, 3, 'static_bare_SMPL'))
        self.assertEqual((p['contact_plane_radius_mm'], p['contact_plane_diameter_mm']), (23, 46))
        self.assertFalse(e['geometry_reporting_correction']['physical_gate_changed'])
        self.assertEqual(e['geometry_reporting_correction']['historical_field'], 'Source32_disk_diameter_mm_23')
        self.assertIn('radius is 23 mm (diameter 46 mm)', self.html)
        self.assertTrue(p['release_and_home_required'])
        a = e['illustrative_L3_physics']
        self.assertAlmostEqual(a['normal_contact_peak_N'], 4.0273328237466135)
        self.assertAlmostEqual(a['Hall_max_N'], 3.988100062117734)
        self.assertEqual((a['combined_force_and_plane_hold_s'], a['forbidden_force_N']), (3.0, 0))
        self.assertLess(a['maximum_hold_tilt_deg'], p['disk_plane_tilt_max_deg'])
        self.assertTrue(a['release_and_home_passed'])

    def test_actual_learned_video_and_poster_are_bound_to_complete_bytes(self):
        e = self.evidence()
        v, p = e['verified_media']['video'], e['verified_media']['poster']
        self.assertEqual((v['bytes'], v['sha256']), (646815, 'b7612f429988c4c3378df1b837f3fcf29998c4bfbfb453fdab70044887c7e456'))
        self.assertEqual((v['frames'], v['fps'], v['encoded_s'], v['recorded_size']), (940, 10, 94.0, [640, 360]))
        self.assertAlmostEqual(v['native_simulation_s'], 94.04583333333333)
        self.assertEqual((v['native_physics_ticks'], v['trailing_unrendered_native_ticks']), (22571, 11))
        self.assertEqual((v['body'], v['split'], v['target']), ('p001120', 'TRAIN', 'L3'))
        self.assertEqual(v['role'], 'actual_ACT9_learned_L2_then_analytic_Hall_L3')
        self.assertEqual(v.get('selected_model_sha256'), e['selected_model']['sha256'])
        self.assertEqual(v.get('source_commit'), e['source_commit'])
        self.assertTrue(v['full_decode_passed'] and v['unedited'])
        self.assertFalse(v['scene_rerendered'] or v['audio'] or v['expert_L2_fallback'])
        for asset in [v, p, e['verified_media']['known_development_matrix']]:
            raw = (DRAFT / asset['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']))
        self.assertEqual(p['source_video_sha256'], v['sha256'])
        self.assertEqual((p['encoded_frame'], p['encoded_time_s']), (500, 50.0))
        self.assertEqual(struct.unpack('>II', (DRAFT / p['file']).read_bytes()[16:24]), (640, 360))
        video = next(a for t, a in Elements(self.html).elements if t == 'video' and a.get('id') == 'iteration9-video')
        self.assertIn('controls', video)
        self.assertEqual(video['preload'], 'metadata')
        self.assertEqual(video['poster'], p['file'])
        self.assertIn('actual ACT9 learned', video['aria-label'])
        self.assertNotIn('autoplay', video)
        matrix = e['verified_media']['known_development_matrix']
        img = next(a for t, a in Elements(self.html).elements if t == 'img' and a.get('src') == matrix['file'])
        self.assertEqual((int(img['width']), int(img['height'])), struct.unpack('>II', (DRAFT / matrix['file']).read_bytes()[16:24]))
        self.assertIn('selection VAL p000923 fails both', img['alt'])
        old = json.loads((DRAFT / 'evidence/act-recovery-measured-teacher-20261107.public.json').read_text())
        self.assertFalse(old['verified_media']['video']['ACT9_inference'])
        self.assertNotEqual(old['verified_media']['video']['sha256'], v['sha256'])

    def test_concise_section_follows_the_dated_historical_snapshot(self):
        self.assertIn('id="iteration9"', self.html, 'Add the closed result after Source32, before evaluation.')
        self.assertLess(self.html.index('id="measured-recovery"'), self.html.index('id="iteration9"'))
        self.assertLess(self.html.index('id="iteration9"'), self.html.index('id="evaluation"'))
        section = self.html.split('id="iteration9"', 1)[1].split('</section>', 1)[0]
        text = re.sub(r'<[^>]+>', ' ', section)
        self.assertLessEqual(len(text.split()), 500)
        self.assertIn('new known TRAIN L3 completion', section)
        self.assertIn('2/4', section)
        self.assertIn('not accepted', section)
        self.assertIn('Historical snapshot · November 7, before development closed', self.html)
        hero = self.html.split('<header class="exp-hero', 1)[1].split('</header>', 1)[0]
        self.assertIn('2/4', hero)
        self.assertNotIn('evaluation is running', hero)
        self.assertTrue(any(a.get('href') == '#iteration9' for _, a in Elements(self.html).elements))

    def test_compact_public_ledger_has_hashes_without_private_paths_or_raw_arrays(self):
        e = self.evidence()
        self.assertEqual(e['source_commit'], 'ac2bc918f733e25f419d73bc83894be7e4d01545')
        self.assertEqual(e['source_artifact_digests']['closed_known_development'], {'bytes': 50761, 'sha256': '8811970c7f75126e5af7a9598c5c29dff0395f616f7103d7eaf483f5157bd195'})
        text = LEDGER.read_text()
        self.assertLess(len(text), 14000)
        for private in ['/Users/', '/opt/', '/workspace/', 'ssh', 'credential', 'actual_fresh_expert', '.npz', '.npy']:
            self.assertNotIn(private, text)
        self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {'states', 'actions', 'joint_positions', 'raw_arrays', 'weights', 'environment'})
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)
        check_keys(e)


if __name__ == '__main__':
    unittest.main()
