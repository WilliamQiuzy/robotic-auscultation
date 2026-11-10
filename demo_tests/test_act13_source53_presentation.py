"""Source53 closed control preserves history and the original negative denominator."""
import hashlib
import json
from pathlib import Path
import re
import unittest

DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration13-source53-balanced'


class Source53Presentation(unittest.TestCase):
    def test_one_append_preserves_the_exact_source52_article(self):
        html = (DRAFT / 'act-recovery-20261106.html').read_text()
        html = re.sub(r'<section\b[^>]*\bid="iteration14-source59-near-fit".*?</section>\n', '', html, flags=re.S)
        # A later, separately tested append must not alter this historical snapshot.
        html = re.sub(r'<section\b[^>]*\bid="iteration13-source56-expert-data".*?</section>\n', '', html, flags=re.S)
        html = re.sub(r'<section\b[^>]*\bid="iteration13-source58-l3-repair".*?</section>\n', '', html, flags=re.S)
        sections = re.findall(r'<section\b[^>]*\bid="' + ENTRY + r'".*?</section>\n', html, re.S)
        self.assertEqual(len(sections), 1)
        self.assertEqual(hashlib.sha256(html.replace(sections[0], '', 1).encode()).hexdigest(),
                         '55977b1fb084905f48008faab63cde042bacc6f83b191791289bcb8ed345aaa0')
        self.assertLess(html.index('id="iteration12-v10-postfit"'), html.index('id="' + ENTRY + '"'))
        self.assertLess(html.index('id="' + ENTRY + '"'), html.index('id="evaluation"'))

    def test_closed_physical_results_retain_all_four_and_actual_media(self):
        p = DRAFT / 'evidence/act13-source53-balanced-20261109.public.json'
        self.assertTrue(p.is_file())
        e = json.loads(p.read_text())
        self.assertEqual((e['physical']['status'], e['physical']['denominator']), ('closed', 4))
        self.assertEqual(e['physical']['passed'], 2)
        self.assertFalse(e['physical']['model_qualified'])
        self.assertEqual(len(e['physical']['cases']), 4)
        for c in e['physical']['cases']:
            self.assertEqual(c['qualified_case_pass'], c['split'] == 'TRAIN')
            self.assertEqual(c['physical']['continuous_force_and_plane_hold_s'], 3 if c['split'] == 'TRAIN' else 0)
            if c['split'] != 'TRAIN':
                self.assertIsNone(c['physical']['maximum_hold_tilt_deg'])
                self.assertIn('120 second deadline', c['failure_reason'])
            self.assertTrue(c['media']['full_decode_passed'])
            for key in ['video', 'poster']:
                a = c['media'][key]
                raw = (DRAFT / a['file']).read_bytes()
                self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (a['bytes'], a['sha256']))
        self.assertEqual(e['training']['new_optimizer_steps'], 1000)
        self.assertFalse(e['training']['optimizer_resumed'])
        self.assertEqual(sum(x['fine'] for x in e['pool']['strata']), 186)
        self.assertEqual(sum(x['recovery'] for x in e['pool']['strata']), 1138)
        self.assertEqual(sum(x['count'] for x in e['sampling']['actual_strata']), 8000)
        self.assertEqual(e['pool']['new_demonstrations'], 0)
        self.assertEqual(len({x['body'] for x in e['pool']['strata']}), 6)
        self.assertEqual(e['pool'].get('origin'), 'native Isaac simulation expert-demonstration windows')
        self.assertEqual(e['pool'].get('identity_context'), 'six HuMMan-derived TRAIN body identities; not six real-patient robot experiments')
        self.assertEqual(e['scope'].get('recordings_context'), 'actual learned-policy Isaac simulation recordings; not physical robot recordings')
        self.assertFalse(e['scope']['fresh_opened'])
        for token in ['/Users/', '/opt/', '/workspace/', '.npz', '.npy', 'row_indices']:
            self.assertNotIn(token, p.read_text())


if __name__ == '__main__':
    unittest.main()
