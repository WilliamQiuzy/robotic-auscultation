"""Published claims stay bound to all four actual cases and scoped metrics."""
import hashlib
import json
from pathlib import Path
import re
import struct
import unittest

D = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration14-source59-near-fit'


class Source59EvidencePresentation(unittest.TestCase):
    def setUp(self):
        self.e = json.loads((D / 'evidence/act14-source59-near-20261110.public.json').read_bytes())
        self.html = (D / 'act-recovery-20261106.html').read_text()

    def test_append_preserves_entire_previously_approved_article(self):
        sections = re.findall(r'<section\b[^>]*\bid="'+ENTRY+r'".*?</section>\n', self.html, re.S)
        self.assertEqual(len(sections), 1)
        html = re.sub(r'<section\b[^>]*\bid="iteration16-source62-field".*?</section>\n', '', self.html, flags=re.S)
        prior = html.replace(sections[0], '', 1).encode()
        self.assertEqual(hashlib.sha256(prior).hexdigest(), self.e['historical_article_sha256_before_append'])
        self.assertEqual(hashlib.sha256(prior).hexdigest(), '02d28d8418c8e7cb844a1280edcd701fc9d6139e06648d434beb1d1b1b6594ef')
        self.assertLess(self.html.index('id="iteration13-source58-l3-repair"'), self.html.index('id="'+ENTRY+'"'))
        self.assertLess(self.html.index('id="'+ENTRY+'"'), self.html.index('id="evaluation"'))

    def test_denominators_modality_and_unrun_control_are_explicit(self):
        e = self.e
        self.assertEqual((e['physical']['passed'], e['physical']['executed'], e['physical']['denominator']), (3,4,4))
        self.assertEqual((e['physical']['train_passed'], e['physical']['development_validation_passed']), (2,1))
        self.assertFalse(e['physical']['model_qualified'])
        self.assertEqual(e['scope']['fresh_cases_tested'], 0)
        self.assertIsNone(e['scope']['fresh_success_rate'])
        for key in ('neural_rgbd_input','neural_hall_input','acoustic_input','fresh_opened','tenpoint_model_qualified','real_hardware_validated'):
            self.assertFalse(e['scope'][key])
        self.assertTrue(e['scope']['prescribed_3d_reference'])
        self.assertEqual((e['training']['treatment_new_optimizer_steps'],e['training']['control_new_optimizer_steps']), (1000,1000))
        self.assertFalse(e['training']['control_physical_evaluated'])
        self.assertFalse(e['training']['validation_used_for_selection'])
        self.assertEqual((e['data']['new_actual_expert_episodes'],e['data']['train_body_identities'],e['data']['pose_templates']), (12,6,1))
        self.assertEqual(e['data']['fine']+e['data']['recovery'], e['data']['overlapping_correcting_e10_windows'])
        self.assertEqual(e['data']['overlapping_correcting_e10_windows'],3000)
        self.assertFalse(e['data']['original_fullbank_cli_exit_receipt_recovered'])
        self.assertTrue(e['data']['subsequent_original_cold_replay_and_official_admission_passed'])

    def test_all_actual_recordings_and_failed_case_remain_accessible(self):
        cases = self.e['physical']['cases']
        self.assertEqual([c['id'] for c in cases], ['p001120-r3','p001120-l3','p000923-r3','p000923-l3'])
        self.assertEqual([c['qualified_case_pass'] for c in cases], [True,True,True,False])
        for c in cases:
            m=c['media'];self.assertTrue(m['full_decode_passed']);self.assertTrue(m['original_video_bytes_preserved'])
            for key in ('video','poster'):
                ref=m[key];raw=(D/ref['file']).read_bytes()
                self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(ref['bytes'],ref['sha256']))
            self.assertIn(m['video']['file'],self.html)
            self.assertEqual((m['width'],m['height'],m['fps']),(640,360,10))
            self.assertEqual(m['first_native_physics_row'],23)
            self.assertEqual(m['last_native_physics_row'],(m['frames']-1)*24+23)
            self.assertEqual(c['runtime']['physics_steps'],m['last_native_physics_row']+1+m['native_ticks_after_last_frame'])
            self.assertTrue(c['worker_normal'] and c['cpu_normal'])
            self.assertFalse(c['infrastructure_error'] or c['expert_l2_fallback'])
        self.assertEqual(cases[-1]['physical']['contact_force_peak_N'],0)
        self.assertEqual(cases[-1]['physical']['continuous_force_and_plane_hold_s'],0)
        self.assertIn('120 second deadline',cases[-1]['failure_reason'])
        for c in cases[:3]:
            p=c['physical'];self.assertEqual(p['continuous_force_and_plane_hold_s'],3)
            self.assertLessEqual(p['maximum_hold_tilt_deg'],3)
            for key in ('lip_N','forbidden_N','frame_N','carriage_other_N','contact_rim_N'):
                self.assertEqual(p['canonical_maximum_force_N'][key],0)
            self.assertGreater(p['canonical_normal_release_confirm_ticks'],0)
            self.assertGreater(p['canonical_home_settle_ticks'],0)

    def test_masked_fit_failure_chart_and_privacy_boundaries(self):
        e=self.e;rows=e['offline']['rows']
        self.assertEqual([r['rows'] for r in rows],[3000,1500,1500])
        self.assertEqual([r['fully_padded_rows'] for r in rows],[0,0,0])
        self.assertGreater(rows[-1]['e10_mae_mrad'],rows[-1]['e10_hold_mae_mrad'])
        self.assertAlmostEqual(rows[-1]['e10_mae_mrad'],1.2626995255014577)
        self.assertEqual(e['offline']['new_optimizer_steps'],0)
        self.assertFalse(e['offline']['physical_success_credit'])
        for ref in e['plot'].values():
            raw=(D/ref['file']).read_bytes();self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(ref['bytes'],ref['sha256']))
        self.assertEqual(struct.unpack('>II',(D/e['plot']['png']['file']).read_bytes()[16:24]),(2128,1088))
        text=(D/'evidence/act14-source59-near-20261110.public.json').read_text()
        for token in ('/Users/','/opt/','/workspace/','.npz','.npy','private_key'):
            self.assertNotIn(token,text)
        self.assertIsNone(e['cost']['whole_owner_wall_s'])
        self.assertAlmostEqual(e['cost']['total_native_simulation_s'],sum(c['runtime']['physics_steps']/240 for c in e['physical']['cases']))


if __name__=='__main__':
    unittest.main()
