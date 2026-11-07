"""ACT3 publication boundaries from supplied closed scalar/media evidence; CPU only."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
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
        self.assertIn('separate ACT4 treatment is running', self.html)
        self.assertIn('No ACT4 training or DEV outcome is reported here', self.html)

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


if __name__ == '__main__':
    unittest.main()
