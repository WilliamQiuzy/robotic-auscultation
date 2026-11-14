"""Check paired interpretation and authenticity of the recorded ACT17 case."""
import hashlib
import json
from pathlib import Path
import re
import unittest


DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration17-source99-approach-feedback'
EVIDENCE = 'evidence/act17-l3-approach-feedback-20261113.public.json'


class ACT17ApproachFeedbackPresentation(unittest.TestCase):
    def test_exact_append_and_recorded_physical_gates(self):
        article = (DRAFT/'act-recovery-20261106.html').read_bytes()
        section, = re.findall(rb'<section\b[^>]*\bid="' + ENTRY.encode() + rb'".*?</section>\n', article, re.S)
        unchanged_article = article.replace(section, b'', 1)
        self.assertEqual(hashlib.sha256(unchanged_article).hexdigest(), 'ffc4880c7e2e341fb1a58b8c5f54545eea04a0e87791357784ae2a4443892815')
        self.assertIn(EVIDENCE.encode(), section)
        self.assertIn(b'4/4', section)
        self.assertIn('ACT 负责远段接近'.encode(), section)
        report = json.loads((DRAFT/EVIDENCE).read_bytes())
        self.assertEqual(report['completed_optimization_steps'], 4000)
        self.assertTrue(report['equivalent_tensor_bits_to_source98'])
        self.assertEqual((report['known4']['executed'], report['known4']['qualified']), (4, 4))
        for case in report['known4']['rows']:
            self.assertGreaterEqual(case['hold_s'], 3.)
            self.assertEqual(case['forbidden_N'], 0.)
            self.assertLessEqual(case['cpu_eof_s'], 2.)
            self.assertGreater(case['home_settle_ticks'], 0)
        video = report['known4_video']
        self.assertEqual((video['frames'], video['fps']), (1052, 10))
        self.assertEqual(hashlib.sha256((DRAFT/'assets'/video['asset']).read_bytes()).hexdigest(), video['sha256'])

    def test_held_out_video_pairs_do_not_claim_unfinished_batch(self):
        report = json.loads((DRAFT/EVIDENCE).read_bytes())
        fresh = report['fresh32']
        self.assertEqual(fresh['status'], 'infrastructure_interruption_no_terminal')
        self.assertEqual((fresh['planned_cases'], fresh['cpu_prepared_cases']), (32, 30))
        self.assertFalse(fresh['act_acceptance_verified'])
        recordings = fresh['selected_recordings']
        self.assertEqual(len(recordings), 4)
        scenarios = {r['scenario'] for r in recordings}
        self.assertEqual(len(scenarios), 2)
        for scenario in scenarios:
            self.assertEqual({r['arm'] for r in recordings if r['scenario'] == scenario}, {'expert', 'act'})
        for r in recordings:
            self.assertEqual(hashlib.sha256((DRAFT/'assets'/r['asset']).read_bytes()).hexdigest(), r['sha256'])
