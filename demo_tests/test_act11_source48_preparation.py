"""Keep historical evidence, pipeline admission and pending physics distinct."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/drafts'
ENTRY = 'iteration11-source48-results'
BASE_ARTICLE_SHA256 = '030d97330a034f81a4e996b7e52a6355b05f9c5428a106da430167f4872e7efe'


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


class ACT11Source48Preparation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()
        self.section = re.search(r'<section\b[^>]*\bid="' + ENTRY + r'".*?</section>\n', self.html, re.S).group(0)
        self.report = json.loads((DRAFT / 'evidence/act11-source48-results-preparation-20261108.public.json').read_text())

    def test_entire_previously_approved_article_is_preserved(self):
        historical = self.html.replace(self.section, '', 1)
        self.assertEqual(hashlib.sha256(historical.encode()).hexdigest(), BASE_ARTICLE_SHA256)
        self.assertEqual(self.html.count('id="' + ENTRY + '"'), 1)
        self.assertLess(self.html.index('id="iteration11-source45-data"'), self.html.index('id="' + ENTRY + '"'))
        self.assertLess(self.html.index('id="' + ENTRY + '"'), self.html.index('id="evaluation"'))

    def test_historical_denominators_close_without_becoming_ACT11_results(self):
        h = self.report['historical_matched8']
        self.assertTrue(h['historical_only'])
        self.assertFalse(h['new_ACT11_or_fresh32_evaluation'])
        self.assertEqual((h['cohort']['subject_count'], h['cohort']['fixed_scenarios_per_arm'], h['cohort']['total_planned_arm_slots']), (2, 8, 24))
        self.assertEqual(len({c['scenario_id'] for c in h['cases']}), 8)
        for arm, key, passed in [('expert', 'expert_qualified', 8), ('act', 'old_ACT_qualified', 0), ('astra', 'Astra_qualified', 8)]:
            a = h['arms'][arm]
            self.assertEqual((a['native_episodes'], a['qualified'], a['failed'], a['unknown']), (8, passed, 8 - passed, 0))
            self.assertEqual(sum(c[key] for c in h['cases']), passed)
            self.assertTrue(a['owner_normal_drained_released'])
        self.assertEqual(h['old_ACT_guard_failures'], {'original_tick_mesh_clearance': 2, 'posture_joint_envelope': 6})
        self.assertEqual((h['Astra']['actual_calls'], h['Astra']['completed_local_skills']), (24, 24))
        self.assertFalse(h['Astra']['learned_motor_policy'] or h['Astra']['causal_modality_effect'] or h['controlled_speed_comparison'])

    def test_pending_physics_has_no_invented_rate_or_success_count(self):
        act = self.report['ACT11_V9']
        self.assertEqual((act['new_optimizer_updates'], act['evaluated_candidates'], act['selected_update']), (20000, 40, 9000))
        known = act['known4']
        self.assertEqual(known['declared_cases'], 4)
        self.assertEqual(known['status'], 'PENDING_TERMINAL_PHYSICAL_EVIDENCE')
        for key in ['completed_cases', 'qualified_cases', 'physical_success_rate']:
            self.assertIsNone(known[key])
        self.assertFalse(known['model_accepted'] or known['new_model_video_available'])
        fresh = act['fresh32']
        self.assertEqual((fresh['identities'] * fresh['known_poses'] * len(fresh['targets']), fresh['declared_scenarios'], fresh['declared_arm_slots']), (32, 32, 64))
        self.assertEqual((fresh['status'], fresh['tested_cases']), ('GATED_NOT_MEASURED', 0))
        self.assertIsNone(fresh['physical_success_rate'])
        self.assertFalse(fresh['payloads_inspected_for_this_draft'])
        pipeline = self.report['pipeline_followup']
        self.assertTrue(pipeline['actual_Linux_proc_and_exec_parity_passed'] and pipeline['fresh_all40_canonical_producer_admission_passed'])
        self.assertFalse(pipeline['selected_weights_changed'] or pipeline['thresholds_changed'] or pipeline['new_training'] or pipeline['model_qualified'])

    def test_small_public_report_and_text_section_add_no_fake_media(self):
        serialized = json.dumps(self.report)
        self.assertLess(len(serialized.encode()), 12000)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'credential', 'ssh']:
            self.assertNotIn(private, serialized)
        self.assertEqual(self.report['publication_state'], 'LOCAL_DRAFT_NOT_PUBLISHED')
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in self.report['source_summary_digests'].values()))
        elements = Elements(self.section).items
        self.assertFalse(any(tag in ['video', 'source', 'img', 'canvas', 'iframe'] for tag, _ in elements))
        for tag, attrs in elements:
            if attrs.get('href'):
                self.assertTrue((DRAFT / attrs['href']).is_file())
        self.assertIn('未测量不是 0/32', self.section)
        self.assertIn('高层技能选择', self.section)
        self.assertIn('104.65 秒采集视频保留其专家干预标签', self.section)


if __name__ == '__main__':
    unittest.main()
