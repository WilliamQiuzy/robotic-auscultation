"""Keep historical evidence, closed development results and unopened Fresh distinct."""
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
        v10_id = 'iteration12-v10-postfit'
        v10 = re.findall(r'<section\b[^>]*\bid="' + v10_id + r'".*?</section>\n', self.html, re.S)
        self.assertEqual(len(v10), 1, 'V10 must remain one explicitly approved append')
        ids = [attrs.get('id') for _, attrs in Elements(self.html).items]
        self.assertEqual(ids.count(v10_id), 1)
        self.assertLess(self.html.index('id="' + ENTRY + '"'), self.html.index('id="' + v10_id + '"'))
        self.assertLess(self.html.index('id="' + v10_id + '"'), self.html.index('id="evaluation"'))
        v11 = re.findall(r'<section\b[^>]*\bid="iteration13-source53-balanced".*?</section>\n', self.html, re.S)
        self.assertEqual(len(v11), 1, 'Source53 must remain one separate append')
        source56 = re.findall(r'<section\b[^>]*\bid="iteration13-source56-expert-data".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source56), 1, 'Source56 must remain one separate append')
        source59 = re.findall(r'<section\b[^>]*\bid="iteration14-source59-near-fit".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source59), 1)
        source62 = re.findall(r'<section\b[^>]*\bid="iteration16-source62-field".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source62), 1)
        source58 = re.findall(r'<section\b[^>]*\bid="iteration13-source58-l3-repair".*?</section>\n', self.html, re.S)
        self.assertEqual(len(source58), 1, 'Source58 must remain one separate append')
        historical = self.html.replace(self.section, '', 1).replace(v10[0], '', 1).replace(v11[0], '', 1).replace(source56[0], '', 1).replace(source58[0], '', 1).replace(source59[0], '', 1).replace(source62[0], '', 1)
        historical = re.sub(r'<section\b[^>]*\bid="iteration17-source67-known4".*?</section>\n', '', historical, flags=re.S)
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

    def test_closed_known4_keeps_original_failures_and_Fresh_unopened(self):
        act = self.report['ACT11_V9']
        self.assertEqual((act['new_optimizer_updates'], act['evaluated_candidates'], act['selected_update']), (20000, 40, 9000))
        known = act['known4']
        self.assertEqual(known['declared_cases'], 4)
        self.assertEqual(known['status'], 'CLOSED_KNOWN_DEVELOPMENT_2_OF_4_NOT_ACCEPTED')
        self.assertEqual((known['completed_cases'], known['qualified_cases'], known['physical_success_rate']), (4, 2, 0.5))
        self.assertFalse(known['model_accepted'])
        self.assertTrue(known['new_model_video_available'])
        self.assertEqual((known['native_closed_case_reports'], known['native_closed_positive_case_reports'], known['native_closed_negative_case_reports']), (4, 2, 2))
        observed = known['case_observations']
        self.assertEqual(len({(c['subject_id'], c['target']) for c in observed}), 4)
        self.assertEqual(sum(c.get('native_reported_success') is True for c in observed), 2)
        self.assertEqual(sum(c.get('native_reported_success') is False for c in observed), 2)
        self.assertFalse(any(c['native_terminal_status'] == 'RUNNING' for c in observed))
        self.assertEqual((observed[0]['role'], observed[0]['subject_id'], observed[0]['target']), ('Known TRAIN', 'p001120', 'R3'))
        self.assertTrue(observed[0]['native_reported_success'] and observed[0]['native_reported_physical_validation'])
        self.assertEqual(observed[0]['independent_qualification'], 'QUALIFIED')
        fresh = act['fresh32']
        self.assertEqual((fresh['identities'] * fresh['known_poses'] * len(fresh['targets']), fresh['declared_scenarios'], fresh['declared_arm_slots']), (32, 32, 64))
        self.assertEqual((fresh['status'], fresh['tested_cases']), ('BLOCKED_NOT_OPENED', 0))
        self.assertIsNone(fresh['physical_success_rate'])
        self.assertFalse(fresh['payloads_inspected_for_this_draft'])
        self.assertTrue(fresh['blocked_by_known_development_failure'])
        self.assertFalse(fresh['gate_open'])
        failure = json.loads((DRAFT / 'evidence/act11-source48-selection-r3-negative-20261109.public.json').read_text())
        self.assertFalse(failure['case']['native_reported_success'] or failure['case']['STOP_entered'] or failure['case']['ALIGN_entered'] or failure['case']['L3_entered'])
        self.assertEqual((failure['case']['actual_physics_steps'], failure['case']['genuine_model_queries']), (29041, 241))
        terminal = failure['streamed_diagnostic_reported']['terminal']
        limits = failure['streamed_diagnostic_reported']['original_interior_requirements']
        self.assertGreater(terminal['observed_tangent_mm'], limits['tangent_mm_at_most'])
        self.assertFalse(failure['Fresh32']['gate_open'] or failure['next_treatment']['new_training_result'])
        self.assertEqual(failure['next_treatment']['status'], 'UNTRAINED_NO_CLOSED_LOOP_RESULT')
        final = json.loads((DRAFT / 'evidence' / known['final_evidence_file']).read_text())
        self.assertEqual((final['fixed_cases'], final['executed_cases'], final['qualified_cases'], final['failed_cases'], final['infrastructure_failures']), (4, 4, 2, 2, 0))
        self.assertEqual(sum(c['original_qualification_passed'] for c in final['cases']), 2)
        self.assertTrue(all(c['worker_complete'] and c['worker_exit_code'] == c['cpu_child_exit_code'] == 0 and c['cpu_child_reaped'] for c in final['cases']))
        self.assertFalse(final['model_accepted'] or final['Fresh32']['gate_open'])
        self.assertIsNone(final['Fresh32']['physical_success_rate'])
        pipeline = self.report['pipeline_followup']
        self.assertTrue(pipeline['actual_Linux_proc_and_exec_parity_passed'] and pipeline['fresh_all40_canonical_producer_admission_passed'])
        self.assertFalse(pipeline['selected_weights_changed'] or pipeline['thresholds_changed'] or pipeline['new_training'] or pipeline['model_qualified'])

    def test_small_public_report_and_actual_media_keep_scope(self):
        serialized = json.dumps(self.report)
        self.assertLess(len(serialized.encode()), 12000)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'credential', 'ssh']:
            self.assertNotIn(private, serialized)
        # This is the report's creation snapshot, not a live publication-status query.
        self.assertEqual(self.report['publication_state'], 'LOCAL_DRAFT_NOT_PUBLISHED')
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in self.report['source_summary_digests'].values()))
        elements = Elements(self.section).items
        self.assertEqual(sum(tag == 'video' for tag, _ in elements), 2)
        self.assertEqual(sum(tag == 'img' for tag, _ in elements), 2)
        self.assertFalse(any(tag in ['canvas', 'iframe'] for tag, _ in elements))
        for tag, attrs in elements:
            for key in ['href', 'src', 'poster']:
                if attrs.get(key):
                    ref = attrs[key]
                    if key == 'href' and ref.startswith('#'):
                        ids = [a.get('id') for _, a in Elements(self.html).items]
                        self.assertEqual(ids.count(ref[1:]), 1, ref)
                    else:
                        self.assertTrue((DRAFT / ref).is_file(), ref)
        self.assertIn('不能记作 0/32', self.section)
        self.assertIn('高层技能选择', self.section)
        self.assertIn('104.65 秒采集视频保留其专家干预标签', self.section)

    def test_real_video_hashes_clocks_and_native_report_do_not_accept_whole_batch(self):
        evidence = json.loads((DRAFT / 'evidence/act11-source48-known-train-r3-video-20261109.public.json').read_text())
        case = evidence['case']
        self.assertEqual((case['role'], case['genuine_model_queries']), ('Known TRAIN development', 50))
        self.assertTrue(case['actual_learned_policy'] and case['native_reported_success'] and case['native_reported_physical_validation'])
        self.assertFalse(case['expert_L2_fallback'] or case['model_qualified'])
        self.assertEqual((case['worker_exit_code'], case['native_reported_validation_errors'], case['native_terminal_status']), (0, [], 'COMPLETE'))
        self.assertEqual(case['independent_whole_Known4_qualification'], 'CLOSED_2_OF_4_MODEL_NOT_ACCEPTED')
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
        failure = json.loads((DRAFT / 'evidence/act11-source48-selection-r3-negative-20261109.public.json').read_text())
        for asset in [failure['media']['video'], failure['media']['poster'], failure['pose_error_figure']]:
            raw = (DRAFT / asset['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']))
        video = failure['media']['video']
        self.assertEqual((video['frames'], video['fps']), (1210, 10))
        self.assertAlmostEqual(video['duration_s'], video['frames'] / video['fps'])
        self.assertEqual(failure['media']['trailing_unrendered_ticks'], 1)
        serialized = json.dumps(failure)
        for private in ['/Users/', '/opt/', '/workspace/', '/mnt/', '.npz', '.npy', 'credential', 'ssh']:
            self.assertNotIn(private, serialized)


if __name__ == '__main__':
    unittest.main()
