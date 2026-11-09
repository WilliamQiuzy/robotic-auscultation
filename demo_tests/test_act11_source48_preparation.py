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
        self.assertFalse(known['model_accepted'])
        self.assertTrue(known['new_model_video_available'])
        self.assertEqual(known['native_closed_positive_case_reports'], 1)
        observed = known['case_observations']
        self.assertEqual(len(observed), 1)
        self.assertEqual((observed[0]['role'], observed[0]['subject_id'], observed[0]['target']), ('Known TRAIN', 'p001120', 'R3'))
        self.assertTrue(observed[0]['native_reported_success'] and observed[0]['native_reported_physical_validation'])
        self.assertEqual(observed[0]['independent_qualification'], 'PENDING')
        fresh = act['fresh32']
        self.assertEqual((fresh['identities'] * fresh['known_poses'] * len(fresh['targets']), fresh['declared_scenarios'], fresh['declared_arm_slots']), (32, 32, 64))
        self.assertEqual((fresh['status'], fresh['tested_cases']), ('GATED_NOT_MEASURED', 0))
        self.assertIsNone(fresh['physical_success_rate'])
        self.assertFalse(fresh['payloads_inspected_for_this_draft'])
        pipeline = self.report['pipeline_followup']
        self.assertTrue(pipeline['actual_Linux_proc_and_exec_parity_passed'] and pipeline['fresh_all40_canonical_producer_admission_passed'])
        self.assertFalse(pipeline['selected_weights_changed'] or pipeline['thresholds_changed'] or pipeline['new_training'] or pipeline['model_qualified'])

    def test_small_public_report_and_actual_media_keep_scope(self):
        serialized = json.dumps(self.report)
        self.assertLess(len(serialized.encode()), 12000)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'credential', 'ssh']:
            self.assertNotIn(private, serialized)
        self.assertEqual(self.report['publication_state'], 'LOCAL_DRAFT_NOT_PUBLISHED')
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in self.report['source_summary_digests'].values()))
        elements = Elements(self.section).items
        self.assertEqual(sum(tag == 'video' for tag, _ in elements), 1)
        self.assertEqual(sum(tag == 'img' for tag, _ in elements), 1)
        self.assertFalse(any(tag in ['canvas', 'iframe'] for tag, _ in elements))
        for tag, attrs in elements:
            for key in ['href', 'src', 'poster']:
                if attrs.get(key):
                    self.assertTrue((DRAFT / attrs[key]).is_file())
        self.assertIn('未测量不是 0/32', self.section)
        self.assertIn('高层技能选择', self.section)
        self.assertIn('104.65 秒采集视频保留其专家干预标签', self.section)

    def test_real_video_hashes_clocks_and_native_report_do_not_accept_whole_batch(self):
        evidence = json.loads((DRAFT / 'evidence/act11-source48-known-train-r3-video-20261109.public.json').read_text())
        case = evidence['case']
        self.assertEqual((case['role'], case['genuine_model_queries']), ('Known TRAIN development', 50))
        self.assertTrue(case['actual_learned_policy'] and case['native_reported_success'] and case['native_reported_physical_validation'])
        self.assertFalse(case['expert_L2_fallback'] or case['model_qualified'])
        self.assertEqual((case['worker_exit_code'], case['native_reported_validation_errors'], case['native_terminal_status']), (0, [], 'COMPLETE'))
        self.assertEqual(case['independent_whole_Known4_qualification'], 'PENDING')
        video = evidence['video']; clock = evidence['native_capture_clock']
        self.assertEqual((video['frames'], video['width'], video['height'], video['fps']), (1069, 640, 360, 10))
        self.assertAlmostEqual(video['encoded_s'], video['frames'] / video['fps'])
        self.assertAlmostEqual(clock['simulation_s'], clock['physics_steps'] * clock['physics_dt_s'])
        self.assertEqual(clock['physics_steps'] - clock['last_frame']['physics_row'] - 1, clock['trailing_unrendered_ticks'])
        self.assertEqual(clock['trailing_unrendered_ticks'], 18)
        self.assertTrue(video['all_captured_frames_retained'] and video['full_decode_passed'])
        self.assertFalse(video['real_time_execution_claim'] or video['new_simulation_or_rerender_for_publication'])
        self.assertGreater(evidence['runtime']['probe_wall_s'], evidence['runtime']['controller_wall_s'])
        self.assertFalse(evidence['runtime']['real_time_claim'] or evidence['poster']['precise_HOLD_start_claim'])
        for asset in [video, evidence['poster'], evidence['contact_subset']['figure']]:
            raw = (DRAFT / asset['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']))
        elements = Elements(self.section).items
        attrs = next(a for tag, a in elements if tag == 'video')
        self.assertIn('controls', attrs); self.assertIn('playsinline', attrs)
        self.assertNotIn('autoplay', attrs)
        self.assertEqual((attrs['preload'], attrs['poster']), ('metadata', evidence['poster']['file']))
        self.assertTrue(any('data-recording-retry' in a for _, a in elements))
        self.assertIn('aria-describedby', attrs)
        self.assertIn('已知 TRAIN 场景，非留出测试', self.section)
        self.assertIn('不代表实时执行', self.section)
        serialized = json.dumps(evidence)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'credential', 'ssh']:
            self.assertNotIn(private, serialized)


if __name__ == '__main__':
    unittest.main()
