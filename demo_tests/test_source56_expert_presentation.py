"""The new expert capture keeps model results, native media and history distinct."""
import hashlib
import json
from pathlib import Path
import re
import unittest

DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration13-source56-expert-data'


class Source56ExpertPresentation(unittest.TestCase):
    def test_single_append_preserves_the_exact_merged_source53_article(self):
        html = (DRAFT / 'act-recovery-20261106.html').read_text()
        html = re.sub(r'<section\b[^>]*\bid="iteration16-source62-field".*?</section>\n', '', html, flags=re.S)
        html = re.sub(r'<section\b[^>]*\bid="iteration14-source59-near-fit".*?</section>\n', '', html, flags=re.S)
        html = re.sub(r'<section\b[^>]*\bid="iteration13-source58-l3-repair".*?</section>\n', '', html, flags=re.S)
        sections = re.findall(r'<section\b[^>]*\bid="' + ENTRY + r'".*?</section>\n', html, re.S)
        self.assertEqual(len(sections), 1)
        self.assertEqual(hashlib.sha256(html.replace(sections[0], '', 1).encode()).hexdigest(),
                         '6cea9f8c4e242169043afee4d8024b1386cfda7bcf6d02dc14440d9c08b16dc6')
        self.assertLess(html.index('id="iteration13-source53-balanced"'), html.index('id="' + ENTRY + '"'))
        self.assertLess(html.index('id="' + ENTRY + '"'), html.index('id="evaluation"'))
        self.assertIn('EXPERT', sections[0])
        self.assertIn('非 ACT 推理录像', sections[0])

    def test_single_expert_stratum_does_not_become_a_model_result(self):
        p = DRAFT / 'evidence/act13-source56-expert-recovery-20261109.public.json'
        e = json.loads(p.read_text())
        self.assertEqual(e['run']['core_code_commit'], '7a660720dee2181771f5eb5cfbac3317a8f082b0')
        self.assertTrue(e['run']['original_owner_normal'])
        self.assertTrue(e['run']['strict_attribution_verified'])
        self.assertEqual(e['run']['errors'], [])
        self.assertEqual((e['coverage']['qualified_strata'], e['coverage']['declared_strata']), (1, 12))
        self.assertEqual(sum(s['status'] == 'missing' for s in e['coverage']['strata']), 11)
        self.assertEqual(e['coverage']['fine']['correcting_full_e10_windows'], 49)
        self.assertEqual(e['coverage']['recovery']['correcting_full_e10_windows'], 201)
        self.assertFalse(e['scope']['training_eligible'])
        self.assertEqual(e['scope']['new_optimizer_steps'], 0)
        self.assertFalse(e['scope']['new_act_result'])
        self.assertFalse(e['scope']['fresh_opened'])
        self.assertTrue(e['physical']['physical_validation'])
        self.assertEqual(e['physical']['continuous_force_and_plane_hold_s'], 3.0)
        self.assertFalse(e['physical']['physical_thresholds_changed'])
        self.assertEqual(e['previous_negative']['acquisition_status'], 'failed_restore')
        self.assertEqual(e['previous_negative']['qualified_training_episodes'], 0)
        self.assertEqual(e['cost']['auxiliary_limit_bytes'] + e['cost']['payload_limit_bytes'],
                         e['cost']['total_capture_limit_bytes'])
        for token in ['/Users/', '/opt/', '/workspace/', '.npz', '.npy', 'private_key']:
            self.assertNotIn(token, p.read_text())

    def test_exact_media_and_native_clock_keep_unrendered_ticks(self):
        e = json.loads((DRAFT / 'evidence/act13-source56-expert-recovery-20261109.public.json').read_text())
        m = e['media']
        for key in ['video', 'poster']:
            a = m[key]
            raw = (DRAFT / a['file']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (a['bytes'], a['sha256']))
        self.assertEqual(m['video']['sha256'], 'd7b249402abc86db62106efd543d12eb4fad42afdf80c4d201af9d2154a02947')
        self.assertEqual(m['poster']['sha256'], '81b64c92ab101431799711d767614b923c4b3a61ab2add5178812169a8f2c4dc')
        self.assertEqual(m['poster_frame_index'], 1440)
        self.assertEqual(m['poster_provenance']['original_frame']['physics_row'], 17291)
        self.assertEqual(m['poster_provenance']['original_frame']['time_s'], 72.05000000001664)
        self.assertFalse(m['poster_provenance']['reencoded'])
        self.assertTrue((DRAFT / m['poster_history'][0]['file']).is_file())
        self.assertEqual((m['width'], m['height'], m['fps'], m['frames']), (256, 144, 20, 2360))
        c = m['native_frame_clock']
        self.assertEqual((c['first_native_physics_row'], c['last_native_physics_row']), (11, 28319))
        self.assertEqual(c['native_ticks_after_last_frame'], 6)
        self.assertEqual(c['full_native_steps'], e['cost']['physics_steps'])
        self.assertEqual(e['cost']['physics_steps'] / e['cost']['native_hz'], e['cost']['simulated_s'])
        self.assertEqual((m['frames'] - 1) * c['native_ticks_between_frames'],
                         c['last_native_physics_row'] - c['first_native_physics_row'])
        self.assertTrue(m['no_synthesized_frames'])


if __name__ == '__main__':
    unittest.main()
