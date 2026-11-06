"""Published presentation boundaries, without native execution or invented results."""
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/drafts'


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class MatchedPresentationTests(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'rebuild-20261105.html').read_text()
        self.elements = Elements(self.html).elements

    def test_fixed_table_preserves_every_case_and_system(self):
        template = json.loads((ROOT / 'benchmarks/matched-eight-results-v1.template.json').read_text())
        expected = {(s['scenario_id'], m) for s in template['scenarios'] for m in template['methods']}
        cells = [(a['data-case'], a['data-method']) for t, a in self.elements if 'data-case' in a]
        self.assertEqual(len(cells), 24)
        self.assertEqual(set(cells), expected)
        source = next(a['data-results-source'] for _, a in self.elements if 'data-results-source' in a)
        actual = json.loads((DRAFT / source).read_text())
        self.assertEqual({s['scenario_id'] for s in actual['scenarios']}, {s['scenario_id'] for s in template['scenarios']})
        if source.endswith('.template.json'):
            for scenario in actual['scenarios']:
                for arm in scenario['arms'].values():
                    self.assertEqual(arm['execution_status'], 'pending')
                    self.assertIsNone(arm['physical_success'])
            self.assertFalse((ROOT / 'docs/evidence/rebuilt-matched-20261106-v3.json').exists())
        elif actual['schema'] == 'ausc-matched-closed-arm-extract-v1':
            self.assertEqual(actual['publication_status'], 'draft')
            self.assertFalse(actual['whole_comparison_complete'])
            self.assertEqual(actual['reported_methods'], ['expert', 'act'])
            self.assertFalse((ROOT / 'docs/evidence/rebuilt-matched-20261106-v3.json').exists())
            for scenario in actual['scenarios']:
                self.assertTrue(scenario['arms']['expert']['physical_success'])
                self.assertFalse(scenario['arms']['act']['physical_success'])
                self.assertEqual(scenario['arms']['astra']['execution_status'], 'pending')
                self.assertIsNone(scenario['arms']['astra']['physical_success'])
                for method in ['expert', 'act']:
                    self.assertIsNone(scenario['arms'][method]['available_scalars']['normal_hold_min_N'])
                    self.assertIsNone(scenario['arms'][method]['available_scalars']['normal_hold_max_N'])
        else:
            self.assertEqual((DRAFT / source).resolve(), ROOT / 'docs/evidence/rebuilt-matched-20261106-v3.json')
            self.assertEqual(actual['publication_status'], 'reviewed_for_append')

    def test_pending_act_and_astra_have_no_fake_media_request(self):
        media = [a for t, a in self.elements if t == 'video']
        self.assertGreaterEqual(len(media), 1)
        self.assertEqual(media[0]['src'], 'assets/matched-v3-expert.mp4')
        pending = [a['data-pending-video'] for t, a in self.elements if 'data-pending-video' in a]
        for method in pending:
            self.assertIn(method, ['act', 'astra'])
            self.assertFalse(any(method in a.get('src', '') for a in media))
        self.assertIn('Independently qualified', self.html)

    def test_original_expert_recording_and_preselection_are_hash_bound(self):
        receipt = json.loads((DRAFT / 'evidence/matched-v3-expert-recording.public.json').read_text())
        self.assertTrue(receipt['actual_recording'])
        self.assertFalse(receipt['new_render'])
        self.assertTrue(receipt['qualified_benchmark_result_pending'])
        self.assertTrue(receipt['media_selection']['selected_before_act_astra_outcomes'])
        self.assertEqual(receipt['media_selection']['scenario_id'], 'p001030-p000476_a001231_f0023-r3')
        self.assertEqual(receipt['media_selection_record']['sha256'], '62fdce646ef9e77a66829e0d5331247da0e53a35816ca13f9b25e0340a786409')
        for asset in receipt['public_assets'].values():
            data = (DRAFT / asset['path']).read_bytes()
            self.assertEqual(len(data), asset['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), asset['sha256'])
        serialized = json.dumps(receipt)
        for private in ['/opt/', '/Users/']:
            self.assertNotIn(private, serialized)
        self.assertIsNone(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", serialized))

    def test_act_media_and_mesh_rejections_bind_actual_closed_receipts(self):
        receipt = json.loads((DRAFT / 'evidence/matched-v3-act-recording.public.json').read_text())
        self.assertEqual(receipt['source_record']['sha256'], '81bd9cfb2d8a4057fee6423888c2f1748f6af748d3e367274e30e8281263c22a')
        self.assertFalse(receipt['physical_qualified'])
        self.assertEqual(receipt['model_request_count'], 54)
        for asset in receipt['public_assets'].values():
            data = (DRAFT / asset['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), asset['sha256'])
        partial = json.loads((DRAFT / 'evidence/rebuilt-matched-20261106-v3.partial.json').read_text())
        categories = [s['arms']['act']['physical_category'] for s in partial['scenarios']]
        self.assertEqual(categories.count('mesh_clearance'), 2)
        self.assertEqual(categories.count('posture_joint_envelope'), 6)
        self.assertEqual(sum(s['arms']['act']['model_calls'] for s in partial['scenarios']), 412)
        for scenario in partial['scenarios']:
            arm = scenario['arms']['act']
            if arm['physical_category'] == 'mesh_clearance':
                self.assertTrue(arm['original_tick_integrity']['original_reader_integrity_passed'])
                self.assertTrue(arm['original_tick_integrity']['expected_guard_failure_not_recording_corruption'])
                self.assertIn('neutral_native_audit_finalize', arm['failure_reason'])
                self.assertIn('mesh collision rejected', arm['original_tick_integrity']['last_event_tick_error'])
        post = json.loads((DRAFT / 'evidence/matched-v3-expert-postflight.public.json').read_text())
        self.assertTrue(post['physical_qualified'])
        self.assertEqual(post['source_video']['sha256'], '61891cf1eebe96083e5662242e669e7dd7d1eff9ea417eded194eec75b3e24dc')

    def test_actual_astra_video_does_not_promote_pending_qualification(self):
        receipt = json.loads((DRAFT / 'evidence/matched-v3-astra-recording.public.json').read_text())
        self.assertEqual(receipt['source_record']['sha256'], 'e4ed3709260eb21e8cee157b6b819b77ee444fe4725ad5642424633ae6530df8')
        self.assertTrue(receipt['actual_recording'])
        self.assertFalse(receipt['new_render'])
        self.assertTrue(receipt['original_worker_physical_reported'])
        self.assertTrue(receipt['qualification_pending'])
        self.assertIsNone(receipt['physical_qualified'])
        self.assertEqual(receipt['model_id'], 'gpt-6-astra')
        self.assertEqual(receipt['model_request_count'], 3)
        self.assertEqual(receipt['motor_actor'], 'analytic_GivenContactExpert')
        for asset in receipt['public_assets'].values():
            data = (DRAFT / asset['path']).read_bytes()
            self.assertEqual(len(data), asset['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), asset['sha256'])
        partial = json.loads((DRAFT / 'evidence/rebuilt-matched-20261106-v3.partial.json').read_text())
        for scenario in partial['scenarios']:
            self.assertIsNone(scenario['arms']['astra']['physical_success'])
            self.assertEqual(scenario['arms']['astra']['execution_status'], 'pending')
        selected = next(s for s in partial['scenarios'] if s['scenario_id'] == receipt['scenario_id'])
        self.assertEqual(selected['arms']['astra']['recording']['status'], 'available')

    def test_final_closed_comparison_binds_all_original_verdicts(self):
        final = json.loads((ROOT / 'docs/evidence/rebuilt-matched-20261106-v3.json').read_text())
        self.assertEqual(final['schema'], 'ausc-matched-qualified-results-v2')
        self.assertTrue(final['whole_comparison_complete'])
        self.assertEqual(final['original_controller']['complete']['sha256'], '864c53cd28814e36381460edfc766017a6fdc56662b646905cfb1e946573f748')
        self.assertEqual(final['original_controller']['frozen_evaluation']['sha256'], '46ea38172ae48f6651b35590b8e4b0314bd038dd9ad675734d03788ec95d4696')
        self.assertEqual(final['summary']['qualified_completions'], {'expert': 8, 'act': 0, 'astra': 8})
        self.assertFalse(final['source_context']['formal_acceptance'])
        self.assertFalse(final['source_context']['training_accepted'])
        self.assertEqual(len(final['scenarios']), 8)
        native = positives = 0
        for scenario in final['scenarios']:
            self.assertEqual(set(scenario['arms']), {'expert', 'act', 'astra'})
            for method, arm in scenario['arms'].items():
                self.assertIsInstance(arm['physical_success'], bool)
                self.assertEqual(arm['original_qualification']['passed'], arm['physical_success'])
                self.assertEqual(arm['original_qualification']['authority'], 'original_native_qualifier')
                self.assertTrue(arm['native_source_matches_plan'])
                self.assertTrue(arm['owner_drained_released'])
                self.assertTrue(arm['actual_native_episode'])
                self.assertIsNone(arm['metrics']['normal_hold_min_N'])
                self.assertIsNone(arm['metrics']['normal_hold_max_N'])
                if method != 'expert':
                    self.assertTrue(arm['model_invoked'])
                    self.assertGreater(arm['model_calls'], 0)
                native += arm['actual_native_episode']
                positives += arm['physical_success']
        self.assertEqual((native, positives), (24, 16))
        astra = final['summary']['astra']
        self.assertEqual(astra['actual_decision_records'], 24)
        self.assertEqual(astra['pause_clock_equal_count'], 24)
        self.assertEqual(astra['reported_executed_native_ticks'], 218585)
        self.assertIsNone(astra['http_only_latency_s'])
        self.assertIsNone(astra['api_cost_usd'])
        self.assertIsNone(astra['pricing_source'])
        self.assertEqual(astra['token_usage_totals']['input_tokens'], 656352)
        self.assertEqual(astra['token_usage_totals']['cached_input_tokens'], 29568)
        self.assertFalse(final['system_roles']['sensor_causality_proven'])
        self.assertFalse(final['system_roles']['learned_astra_motor_policy'])

    def test_final_public_artifact_refs_match_bytes_and_hashes(self):
        final = json.loads((ROOT / 'docs/evidence/rebuilt-matched-20261106-v3.json').read_text())
        def walk(value):
            if isinstance(value, dict):
                if value.get('public_path') and 'sha256' in value and 'bytes' in value:
                    target = ROOT / 'docs' / value['public_path']
                    self.assertTrue(target.is_file(), str(target))
                    raw = target.read_bytes()
                    self.assertEqual(len(raw), value['bytes'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), value['sha256'])
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(final)
        self.assertEqual((ROOT / 'benchmarks/matched-qualified-results-v2.schema.json').read_bytes(), (ROOT / 'docs/evidence/matched-qualified-results-v2.schema.json').read_bytes())
        serialized = json.dumps(final)
        self.assertNotIn('/opt/', serialized)
        self.assertNotIn('/Users/', serialized)
        self.assertNotIn('GPU-', serialized)

    def test_article_is_published_closed_and_all_local_targets_exist(self):
        self.assertFalse(any(t == 'meta' and a.get('name') == 'robots' and 'noindex' in a.get('content', '') for t, a in self.elements))
        self.assertTrue(any(t == 'link' and a.get('rel') == 'canonical' and a.get('href') == 'https://williamqiuzy.github.io/robotic-auscultation/drafts/rebuild-20261105.html' for t, a in self.elements))
        home = (ROOT / 'docs/index.html').read_text()
        self.assertIn('id="rebuilt-comparison"', home)
        self.assertIn('Historical release', home)
        ids = {a['id'] for _, a in self.elements if 'id' in a}
        for tag, attrs in self.elements:
            for name in ['href', 'src', 'poster']:
                target = attrs.get(name)
                if not target or target.startswith(('http:', 'https:', 'data:')):
                    continue
                if target.startswith('#'):
                    self.assertIn(target[1:], ids)
                else:
                    self.assertTrue((DRAFT / target.split('#')[0]).is_file(), target)


if __name__ == '__main__':
    unittest.main()
