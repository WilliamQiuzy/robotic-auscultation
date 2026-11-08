"""ACT10 expert data boundaries; no simulator or model execution required."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest


DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
LEDGER = DRAFT / 'evidence/act-recovery-iteration10-expert-data-20261107.public.json'


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


class ACT10ExpertDataPresentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()

    def test_bank_qualification_retains_planned_denominator_and_label_boundary(self):
        e = json.loads(LEDGER.read_text())
        d = e['late_DART_expert_bank']
        self.assertEqual((d['planned'], d['prepared'], d['CPU_plan_unavailable'], d['independently_qualified']), (20, 17, 3, 17))
        self.assertFalse(d['unavailable_proves_geometric_unreachability'])
        self.assertEqual((d['measured_TRAIN_bodies'], d['windows'], d['full_H20_windows'], d['valid_future_cells']), (5, 274, 274, 5480))
        b = d['label_boundary']
        self.assertEqual((b['current_states_after_perturbation'], b['current_states_before_or_during_perturbation'], b['earlier_perturbation_H2_measurements']), (274, 0, 17))
        self.assertFalse(b['earlier_H2_measurements_are_positive_labels'])
        self.assertFalse(d['labels_repaired'] or d['normalization_refitted'] or d['actual_student_failure_data'])
        self.assertTrue(d['bank_qualified'])
        combined = e['combined_measured_training_data']
        self.assertEqual((combined['episodes'], combined['windows'], combined['physical_TRAIN_bodies']), (21, 332, 6))
        self.assertEqual(combined['unchanged_native_origins'], {'TRAIN': 8373, 'selection_VAL': 2345})
        self.assertEqual(combined['computed_auxiliary_TRAIN_bodies'], 16)

    def test_full_pool_diversity_does_not_imply_diverse_handoff_corridor(self):
        e = json.loads(LEDGER.read_text())
        c = e['measured_recovery_corridor']
        self.assertEqual(c['metric'], 'dynamic_prepose_error')
        self.assertEqual(c['threshold_m'], 0.012)
        self.assertEqual((c['full_measured_windows'], c['corridor_measured_windows'], c['corridor_TRAIN_bodies']), (332, 53, 1))
        q, d = c['query32'], c['dart37']
        self.assertEqual((q['corridor_windows'], q['corridor_TRAIN_bodies'], q['R3_corridor_windows'], q['L3_corridor_windows']), (53, 1, 17, 36))
        self.assertEqual((d['full_pool_windows'], d['full_pool_TRAIN_bodies'], d['corridor_windows'], d['corridor_TRAIN_bodies']), (274, 5, 0, 0))
        self.assertEqual((d['slots'], d['slots_without_corridor_windows'], d['R3_full_pool_windows'], d['L3_full_pool_windows']), (17, 17, 164, 110))
        self.assertEqual(q['corridor_windows'] + d['corridor_windows'], c['corridor_measured_windows'])
        self.assertEqual(c['separate_skin_perturbation_trigger_mm'], e['late_DART_expert_bank']['skin_trigger_mm'])
        self.assertEqual(c['separate_skin_perturbation_trigger_mm'], 35)
        self.assertFalse(c['sampler_changed'] or c['selection_changed'] or c['model_queried'])
        ref = e['source_artifact_digests']['measured_corridor_counts']
        self.assertEqual((ref['bytes'], ref['sha256']), (11379, 'f3430ae3bbbe61790c8810de51ec42ea554af4fa22b733646272be53b3498e6a'))
        paragraph = re.search(r'<p\b[^>]*\bid="iteration10-corridor-limit".*?</p>', self.html, re.S).group(0)
        for fact in ['dynamic prepose error ≤12 mm', 'All 53', 'one TRAIN body', '0/274', 'all 17 slots', '35 mm skin trigger']:
            self.assertIn(fact, paragraph)

    def test_preparation_and_started_training_are_not_model_acceptance(self):
        e = json.loads(LEDGER.read_text())
        t = e['ACT10_training_snapshot']
        self.assertTrue(t['CPU_preparation_passed'] and t['official_CPU_integration_passed'] and t['started'])
        self.assertFalse(t['complete'])
        self.assertEqual(t['launched_UTC'], '2026-11-07T23:48:48.446731+00:00')
        self.assertEqual((t['planned_new_updates'], t['evaluation_every_updates'], t['planned_full_native_VAL_evaluations']), (20000, 500, 40))
        self.assertEqual(t['initialization'], {'warm_start': 'ACT9', 'exact_tensors': 54, 'fresh_AdamW_initial_step': 0})
        self.assertEqual((t['known_development_cases'], t['fresh_cases']), (0, 0))
        self.assertFalse(t['model_accepted'])
        section = re.search(r'<section\b[^>]*\bid="iteration10".*?</section>', self.html, re.S).group(0)
        self.assertIn('expert data qualification', section)
        self.assertIn('model is not accepted', section)
        self.assertIn('20 planned', section)
        self.assertIn('CPU plan unavailable', section)
        self.assertLessEqual(len(re.sub(r'<[^>]+>', ' ', section).split()), 650)

    def test_actual_expert_media_bytes_and_semantics_are_bound(self):
        e = json.loads(LEDGER.read_text())
        media = e['illustrative_expert_smoke']
        self.assertFalse(media['learned_ACT_inference'] or media['recording_of_full_17_episode_bank'])
        v, poster = media['video'], media['poster']
        self.assertEqual((v['bytes'], v['sha256']), (639517, '293fe69c55ddb00aa18bee4194c86a3fa24ec5427e7b8f845b7a5dffd04cde81'))
        self.assertEqual((v['frames'], v['fps'], v['recorded_size'], v['encoded_s']), (2193, 20, [256, 144], 109.65))
        for asset in (v, poster):
            raw = (DRAFT / asset['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (asset['bytes'], asset['sha256']))
        video = next(a for tag, a in Elements(self.html).items if tag == 'video' and a.get('id') == 'iteration10-expert-video')
        self.assertIn('controls', video)
        self.assertNotIn('autoplay', video)
        self.assertEqual(video['poster'], poster['file'])
        self.assertIn('actual expert DART recovery', video['aria-label'])
        self.assertTrue(media['full_decode_passed'] and media['source_frame_count_preserved'])

    def test_new_entry_preserves_ACT9_history_order_and_public_privacy(self):
        old = re.search(r'<section\b[^>]*\bid="iteration9".*?</section>', self.html, re.S).group(0)
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(), 'aeae66b7a1b0c7dcebc3c89f049c9d1d2afc59f6e50ded1af39effc55e58a01f')
        self.assertLess(self.html.index('id="iteration9"'), self.html.index('id="iteration10"'))
        self.assertLess(self.html.index('id="iteration10"'), self.html.index('id="evaluation"'))
        self.assertIn('href="#iteration10"', self.html)
        text = LEDGER.read_text()
        self.assertLess(len(text), 12000)
        for private in ['/Users/', '/opt/', '/workspace/', 'ssh', 'credential', '.npz', '.npy', 'gpu_uuid']:
            self.assertNotIn(private, text)
        self.assertIsNone(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {'states', 'actions', 'joint_positions', 'raw_arrays', 'weights', 'environment'})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(json.loads(text))


if __name__ == '__main__':
    unittest.main()
