"""Bind the notebook append to completed training and the scoped CPU veto."""
import hashlib
import json
from pathlib import Path
import re
import struct
import unittest

DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration16-source62-field'
BASE_ARTICLE_SHA256 = '73e89d574c07cc8fbeed1ed6d0d49bb517ea778b40f090bd79970dde2b454970'
EVIDENCE = 'evidence/act16-source62-fixed-field-20261110.public.json'


class Source62FieldPresentation(unittest.TestCase):
    def setUp(self):
        self.html = (DRAFT / 'act-recovery-20261106.html').read_text()
        sections = re.findall(r'<section\b[^>]*\bid="' + ENTRY + r'".*?</section>\n', self.html, re.S)
        self.assertEqual(len(sections), 1)
        self.section = sections[0]
        self.public_json = (DRAFT / EVIDENCE).read_text()
        self.evidence = json.loads(self.public_json)

    def test_append_preserves_the_full_source59_article(self):
        html = re.sub(r'<section\b[^>]*\bid="iteration17-source67-known4".*?</section>\n', '', self.html, flags=re.S)
        prior = html.replace(self.section, '', 1).encode()
        self.assertEqual(hashlib.sha256(prior).hexdigest(), BASE_ARTICLE_SHA256)
        self.assertEqual(self.evidence['historical_article_sha256_before_append'], BASE_ARTICLE_SHA256)
        self.assertLess(self.html.index('id="iteration14-source59-near-fit"'), self.html.index('id="' + ENTRY + '"'))
        self.assertLess(self.html.index('id="' + ENTRY + '"'), self.html.index('id="evaluation"'))
        self.assertIn('href="#iteration14-source59-near-fit"', self.section)
        self.assertIn(f'href="{EVIDENCE}"', self.section)
        self.assertNotIn('<video', self.section)

    def test_capacity_and_paired_training_are_distinct_completed_runs(self):
        capacity = self.evidence['source61_train_capacity']
        self.assertEqual((capacity['train_windows'], capacity['new_optimizer_updates'], capacity['actual_draws']),
                         (3000, 1000, 64000))
        before, after, hold = (capacity['l3_e10_mae_mrad'][key] for key in ('before', 'after', 'hold'))
        self.assertLess(after, hold)
        self.assertLess(hold, before)
        self.assertTrue(capacity['capacity_fit_gate_passed'])
        self.assertFalse(capacity['checkpoint_used_for_source62'])
        for value in (before, after, hold):
            self.assertIn(f'{value:.3f} mrad', self.section)

        pair = self.evidence['source62_paired_training']
        self.assertTrue(pair['same_source59_warm_start'])
        self.assertTrue(pair['same_training_corpus'])
        self.assertTrue(pair['same_base_rng_and_n1n2_draws'])
        self.assertTrue(pair['same_auxiliary_draws'])
        self.assertFalse(pair['validation_checkpoint_selection'])
        for name, near_count in [('near8', 32000), ('near16', 64000)]:
            arm = pair['arms'][name]
            self.assertEqual((arm['new_optimizer_updates'], arm['actual_draws'], arm['near_draws']),
                             (4000, 256000, near_count))
            self.assertGreater(arm['owner_wall_s'], arm['worker_wall_s'])
            self.assertIn(f'{arm["worker_wall_s"]:.2f} s', self.section)
            self.assertIn(f'{arm["owner_wall_s"]:.2f} s', self.section)

    def test_cpu_results_preserve_direction_veto_and_evaluation_scope(self):
        prior = self.evidence['source60_read_only_source59_replay']
        self.assertEqual(prior['new_optimizer_updates'], 0)
        self.assertFalse(prior['causal_conclusion_established'])
        self.assertEqual(prior['target_prepose_offset_mm'], 25)
        self.assertEqual([q['query'] for q in prior['late_queries']], [120, 180, 220, 241])
        self.assertTrue(all(18 < q['prepose_error_mm'] < 20 for q in prior['late_queries']))
        self.assertTrue(all(abs(q['intended_e10_progress_mm']) < 0.02 for q in prior['late_queries']))

        cpu = self.evidence['source62_cpu_field']
        self.assertEqual(cpu['reused_source'], 'Source59 development L3')
        self.assertTrue(cpu['near16_veto'])
        self.assertEqual(cpu['veto_prepose_error_above_mm'], 2)
        self.assertEqual(cpu['veto_progress_at_or_below_mm'], 0.025)
        self.assertLess(cpu['l3_teacher_e10_mae_mrad']['near16'], cpu['l3_teacher_e10_mae_mrad']['near8'])
        for name in ('near8', 'near16'):
            self.assertIn(f'{cpu["l3_teacher_e10_mae_mrad"][name]:.3f} mrad', self.section)
        self.assertEqual([q['query'] for q in cpu['late_queries']], [120, 180, 220, 241])
        for field_q, prior_q in zip(cpu['late_queries'], prior['late_queries']):
            self.assertEqual(field_q['origin_prepose_error_mm'], prior_q['prepose_error_mm'])
            for name in ('near8', 'near16'):
                self.assertLess(field_q['predicted_e10_progress_mm'][name], -0.25)
                self.assertGreater(field_q['predicted_e10_progress_mm'][name], -0.36)
        self.assertEqual(self.evidence['scope'], {
            'new_isaac_cases': 0,
            'new_isaac_recordings': 0,
            'fresh_opened': False,
            'fresh_cases_tested': 0,
            'model_qualified': False,
        })
        for phrase in ('预设的筛选门槛', '没有执行新的 Isaac 回合', '封存新身份仍未开放', '两种坐姿', '多个轴的正负方向'):
            self.assertIn(phrase, self.section)

    def test_public_plot_is_bound_without_private_runtime_metadata(self):
        figure = self.evidence['plot']
        raw = (DRAFT / figure['file']).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (figure['bytes'], figure['sha256']))
        self.assertEqual(struct.unpack('>II', raw[16:24]), (2144, 864))
        self.assertIn(figure['file'], self.section)
        for content in (self.public_json, self.section):
            self.assertNotRegex(content, r'/(?:Users|opt|workspace|private|home)/|p\d{6}\b|\.safetensors\b')
            self.assertNotRegex(content, r'\b(?:\d{1,3}\.){3}\d{1,3}\b')


if __name__ == '__main__':
    unittest.main()
