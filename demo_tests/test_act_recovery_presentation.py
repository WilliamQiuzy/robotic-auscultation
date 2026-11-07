"""ACT recovery publication boundaries from closed scalar/media evidence; CPU only."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/drafts'


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class ACT3Presentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()
        self.evidence = json.loads((DRAFT / 'evidence/act-recovery-iteration3-20261106.public.json').read_text())

    def test_closed_training_is_separate_from_four_native_negatives(self):
        raw = (DRAFT / 'evidence/act-recovery-iteration3-20261106.public.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '9ff83dd3870b711b0396e638f04552c20fa1e9e8b6e2184d46312846259d732b')
        e = self.evidence
        self.assertEqual((e['completed_steps'], e['selected_step']), (20000, 20000))
        self.assertEqual((e['physical_successes'], e['attempted_cases']), (0, 4))
        self.assertEqual(len(e['evaluations']), 40)
        self.assertEqual(len(e['cases']), 4)
        self.assertTrue(all(c['passed'] is False for c in e['cases']))
        self.assertEqual(e['selected_metrics']['validation']['sample_count'], 2345)
        self.assertAlmostEqual(e['selected_metrics']['train']['act']['full_chunk_mae_rad'] * 1000, 6.960135999178431)
        self.assertAlmostEqual(e['selected_metrics']['validation']['act']['full_chunk_mae_rad'] * 1000, 10.099932097573028)
        self.assertTrue(e['state_only'])
        self.assertFalse(e['RGBD_neural_inputs'])
        self.assertFalse(e['Hall_neural_input'])
        self.assertFalse(e['fresh_identities_exposed'])
        self.assertFalse(e['physics_gates_changed'])
        self.assertIn('ACT4 also closed negative', self.html)

    def test_actual_act_video_retains_unrendered_terminal_ticks(self):
        r = self.evidence['recording']
        self.assertTrue(r['unedited'])
        self.assertTrue(r['learned_policy'])
        self.assertFalse(r['expert_l2_fallback'])
        self.assertFalse(r['formal_acceptance'])
        self.assertEqual(r['executed_controller_class'], 'sim.ausc_policy.m2_given_contact_expanded_pilot_deployment.ExpandedGivenACT')
        self.assertEqual((r['frames'], r['width'], r['height'], r['fps']), (145, 640, 360, 10))
        self.assertEqual(r['duration_s'], 14.5)
        self.assertAlmostEqual(r['simulation_s'], 14.55)
        self.assertEqual(r['trailing_unrendered_physics_ticks'], 12)
        movie = (DRAFT / 'assets' / r['file']).read_bytes()
        self.assertEqual(len(movie), r['bytes'])
        self.assertEqual(hashlib.sha256(movie).hexdigest(), r['sha256'])
        self.assertIn('does not visually demonstrate contact or full arm motion', self.html)

    def test_cpu_plot_is_not_physics_or_a_feasibility_certificate(self):
        p = DRAFT / 'evidence/act-recovery-iteration3-posture-20261106.public.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), 'efe251bb4437eb70b62c2058673f0ad6285f2ec2dde5d2d0b93df9bb9a6a60be')
        d = json.loads(p.read_text())
        self.assertFalse(d['physics_executed'])
        self.assertFalse(d['physical_qualification'])
        self.assertFalse(d['model_inference_executed'])
        self.assertFalse(d['first_rejected_tick']['applied'])
        self.assertEqual(d['first_unsafe_raw_target']['query_index'], 24)
        self.assertLess(d['first_unsafe_raw_target']['J3_margin_mrad'], 0)
        self.assertGreater(d['query25_26_27_J3_reserve_mrad']['26'], 0)
        self.assertLess(d['query25_26_27_J3_reserve_mrad']['27'], 0)
        self.assertEqual(d['decode_max_error_rad'], 0)
        self.assertIn('is not a motion feasibility certificate', self.html)
        bindings = json.loads((DRAFT / 'evidence/act-recovery-iteration3-bindings-20261106.public.json').read_text())
        self.assertEqual(bindings['references']['independent_reference_verification']['verified_references'], 239)
        for name, asset in bindings['public_files'].items():
            b = (DRAFT / name).read_bytes()
            self.assertEqual(len(b), asset['bytes'])
            self.assertEqual(hashlib.sha256(b).hexdigest(), asset['sha256'])

    def test_secondary_issue_is_not_claimed_as_public_error_text(self):
        case = next(c for c in self.evidence['cases'] if c['id'] == 'p000923-r3')
        self.assertEqual(case['error'], 'ValueError: Actual closed successful learned native result without errors required')
        bindings = json.loads((DRAFT / 'evidence/act-recovery-iteration3-bindings-20261106.public.json').read_text())
        cause = next(c for c in bindings['underlying_dev_failures'] if c['role'] == 'selection_VAL' and c['target'] == 'R3')
        self.assertEqual(cause['reported_minimum_m'], 0.002999369900189177)
        self.assertTrue(cause['secondary_audit_finalize_error_retained'])
        self.assertTrue('the exact clearance scalar is bound in the' in self.html, 'The public clearance binding must be explicit.')
        self.assertTrue('The secondary audit finalization issue is recorded in the original Linux receipts;' in self.html, 'Secondary issue provenance must name the original Linux receipts.')
        self.assertTrue('the public case ledger retains the generic qualification verdict.' in self.html, 'The generic public verdict must not be presented as detailed secondary error text.')
        self.assertNotIn('the exact scalar and secondary audit finalization error are retained in the evidence', self.html)

    def test_local_links_and_privacy_remain_valid_without_publication(self):
        elements = Elements(self.html).elements
        ids = {a['id'] for _, a in elements if 'id' in a}
        self.assertTrue(any(t == 'meta' and a.get('name') == 'robots' and a.get('content') == 'noindex' for t, a in elements))
        for _, attrs in elements:
            for key in ['href', 'src', 'poster']:
                raw = attrs.get(key)
                if not raw:
                    continue
                target = urlsplit(raw)
                if target.scheme or target.netloc:
                    continue
                if target.path:
                    self.assertTrue((DRAFT / unquote(target.path)).is_file(), raw)
                elif target.fragment:
                    self.assertIn(target.fragment, ids)
        for p in DRAFT.glob('evidence/act-recovery-iteration3*.json'):
            serialized = p.read_text()
            self.assertNotIn('/opt/', serialized)
            self.assertNotIn('/Users/', serialized)
            self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', serialized))


class ACT4Presentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()

    def test_current_article_replaces_pending_act4_with_a_closed_negative(self):
        elements = Elements(self.html).elements
        self.assertTrue(any(a.get('id') == 'iteration4' for _, a in elements))
        self.assertTrue(any(a.get('href') == '#iteration4' for _, a in elements))
        self.assertIn('Four treatments closed negative', self.html)
        self.assertNotIn('separate ACT4 result pending', self.html)
        self.assertNotIn('separate ACT4 treatment is running', self.html)
        self.assertIn('handoff record is not a successful handoff', self.html)

    def test_exact_actual_result_retains_the_handoff_and_shadowed_failure_distinction(self):
        p = DRAFT / 'evidence/act-recovery-iteration4-20261107.public.json'
        self.assertTrue(p.is_file(), 'ACT4 must include the exact public result, not a prose-only verdict.')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), '31baa08307b34e36c5752fb4b4bdcaae45f0c5582269e6b34072360dfbdc6c82')
        e = json.loads(p.read_text())
        self.assertEqual((e['completed_steps'], e['selected_step'], len(e['evaluations'])), (20000, 18500, 40))
        self.assertAlmostEqual(e['raw_target_mae_rad']['train'] * 1000, 7.908001314907445)
        self.assertAlmostEqual(e['raw_target_mae_rad']['selection_validation'] * 1000, 11.192285613977882)
        d = e['actual_development']
        self.assertEqual((d['fixed_denominator'], d['physical_successes'], len(d['outcomes'])), (4, 0, 4))
        for c in d['outcomes']:
            self.assertFalse(c['passed'])
            self.assertEqual(c['l2_handoff_record_count'], 1)
            self.assertEqual(c['actual_l2_handoff_successes'], 0)
            self.assertEqual(c['actual_l3_starts'], 0)
        stops = [c['l2_outcomes'][0]['failure_reason'] for c in d['outcomes']]
        self.assertEqual(stops.count('M2 L2 clock/120 second deadline failed'), 3)
        self.assertEqual(stops.count('Stop preview CPU deadline'), 1)
        self.assertEqual(sum('100000000' in c['failure_reason'] for c in d['outcomes']), 3)
        self.assertFalse(e['expert_l2_fallback'])
        self.assertFalse(e['fresh_identities_exposed'])
        self.assertIn('later 100 MB directory cap', self.html)

    def test_media_and_scalars_are_byte_bound_with_derived_poster_provenance(self):
        p = DRAFT / 'evidence/act-recovery-iteration4-bindings-20261107.public.json'
        self.assertTrue(p.is_file(), 'New public files need a downloadable byte/hash manifest.')
        bindings = json.loads(p.read_text())
        for name, asset in bindings['public_files'].items():
            raw = (DRAFT / name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']), name)
        movies = bindings['recordings']
        self.assertEqual([(v['frames'], v['duration_s']) for v in movies], [(1210, 121.0), (167, 16.7)])
        self.assertTrue(all(v['unedited'] and v['learned_policy'] and not v['expert_l2_fallback'] for v in movies))
        poster = bindings['derived_posters'][0]
        self.assertEqual(poster['source_video_sha256'], movies[0]['sha256'])
        self.assertEqual(poster['requested_time_s'], 25)
        self.assertIn('does not establish force or contact', self.html)
        self.assertIn('Scientific CPU diagnostic', self.html)

    def test_new_diagnostic_keeps_its_native_image_aspect_ratio(self):
        for tag, attrs in Elements(self.html).elements:
            if tag == 'img' and 'iteration4' in attrs.get('src', ''):
                native = struct.unpack('>II', (DRAFT / attrs['src']).read_bytes()[16:24])
                self.assertEqual((int(attrs['width']), int(attrs['height'])), native)

    def test_v3_is_preparation_and_preserves_generated_vs_measured_sources(self):
        p = DRAFT / 'evidence/act-recovery-v3-preparation-20261107.public.json'
        self.assertTrue(p.is_file(), 'Prepared CPU work must have a separate bounded summary.')
        e = json.loads(p.read_text())
        self.assertFalse(e['training_started'])
        self.assertFalse(e['physics_executed'])
        self.assertFalse(e['fresh_identities_exposed'])
        self.assertEqual(e['original_failed_states']['cpu_checks_passed'], 12)
        self.assertEqual(e['new_genuine_train_queries']['accepted'], 8)
        g = e['conditional_generated_h2']
        self.assertEqual((g['slots'], g['accepted'], g['rejected'], g['computed_cells']), (1024, 854, 170, 17080))
        self.assertFalse(g['genuine_dart'])
        self.assertEqual(g['full_original_checks'], 208318)
        self.assertEqual(e['preserved_parent_auxiliary']['origins'], 35120)
        self.assertEqual(e['bounded_head']['physical_inset_rad'], .001)
        self.assertEqual(e['bounded_head']['softness_rad'], .010)
        self.assertIn('V3 preparation', self.html)
        self.assertIn('not genuine DART', self.html)
        self.assertIn('not a causal ablation', self.html)
        for p in DRAFT.glob('evidence/act-recovery-*20261107.public.json'):
            raw = p.read_text()
            self.assertNotIn('/Users/', raw)
            self.assertNotIn('/opt/', raw)
            self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', raw))

    def test_original_failed_state_checks_bind_source25_and_only_a_numerical_tail(self):
        e = json.loads((DRAFT / 'evidence/act-recovery-v3-preparation-20261107.public.json').read_text())
        states = e['original_failed_states']
        self.assertEqual(states.get('source_commit'), 'ad5a3ecd958ee6c2e2ee399623232f2414e894b6')
        self.assertIn('numerical stationary tail', states['checks'])
        self.assertNotIn('settling', states['checks'])
        self.assertEqual(states['numerical_stationary_tail'], {'identical_commands': 3, 'constant_tail_ticks': 24})
        for key in ['measured_settling_verified', 'actual_native_preview_live_verified', 'actual_physics_executed',
                    'actual_native_output_restored', 'actual_cumulative_retry_budget_qualified', 'source26_counterfactual_rerun']:
            self.assertIs(states[key], False, key)
        refs = e['original_receipt_references']
        q = refs['source25_cpu_counterfactual']
        self.assertEqual((q['bytes'], q['sha256']), (5428, '42ed58ec0ba36fd0d3687615ad613dab9b01459daf240694fc69918cfd3675d5'))
        manifest = refs['source25_manifest']
        self.assertEqual((manifest['bytes'], manifest['sha256']), (287026, '719349708f434f1b39ec4b1f297c7719b58f76f37bcc96244dd1721c2b7d6c69'))
        self.assertIn('Source25’s CPU counterfactual checks', self.html)
        self.assertNotIn('Source26’s CPU counterfactual checks', self.html)
        self.assertIn('numerical stationary tail', self.html)
        self.assertIn('measured settling', self.html)

    def test_closed_cpu_export_does_not_complete_integration_training_or_physics(self):
        e = json.loads((DRAFT / 'evidence/act-recovery-v3-preparation-20261107.public.json').read_text())
        export = e.get('exported_auxiliary', {})
        self.assertIs(export.get('passed'), True)
        self.assertEqual((export['origins'], export['sources'], export['slots']), (35982, 934, 1109))
        self.assertEqual(export['oracle'], {'origins': 20, 'slots': 21, 'computed_cells': 398})
        self.assertTrue(e['preserved_parent_auxiliary']['preserved_exact_ordered_prefix'])
        self.assertEqual(e['official_cpu_integration']['status'], 'running')
        self.assertFalse(e['training_started'])
        self.assertFalse(e['physics_executed'])
        self.assertNotIn('final export and official integration verification', e['pending'])
        refs = e['original_receipt_references']
        self.assertEqual(refs['exported_data']['sha256'], 'd84afc9f705f25de34bc2aeb253ee79b43a52981ad6fdc57e3f19265da47b075')
        self.assertEqual(refs['export_proof']['sha256'], '5e7fd554b4f631bce866fbd431d20eeaca09ea47136bf075fd44e5cbdb707b66')
        self.assertIn('official CPU integration is running', self.html)


if __name__ == '__main__':
    unittest.main()
