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
        self.assertIn('Astra pending', self.html)

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

    def test_article_is_noindex_partial_and_all_local_targets_exist(self):
        self.assertTrue(any(t == 'meta' and a.get('name') == 'robots' and a.get('content') == 'noindex,nofollow' for t, a in self.elements))
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
