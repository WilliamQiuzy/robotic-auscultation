"""Verify the ACT17 article append and the scope of its actual recordings."""
import hashlib
import json
from pathlib import Path
import re
import unittest


DRAFT = Path(__file__).resolve().parents[1] / 'docs/drafts'
ENTRY = 'iteration17-source67-known4'
EVIDENCE = 'evidence/act17-source67-matched-recovery-20261111.public.json'


class ACT17Source67Presentation(unittest.TestCase):
    def test_append_preserves_the_approved_article(self):
        page = (DRAFT / 'act-recovery-20261106.html').read_bytes()
        page = re.sub(rb'<section\b[^>]*\bid="iteration17-source99-approach-feedback".*?</section>\n', b'', page, flags=re.S)
        sections = re.findall(rb'<section\b[^>]*\bid="' + ENTRY.encode() + rb'".*?</section>\n', page, re.S)
        self.assertEqual(len(sections), 1)
        prior = page.replace(sections[0], b'', 1)
        self.assertEqual(hashlib.sha256(prior).hexdigest(),
                         '35eca7cbe36b14e13d7ed86e11227bfc517f396b216c1072552959dc64c7c1ec')
        self.assertIn(EVIDENCE.encode(), sections[0])

    def test_media_is_bound_to_three_completed_cases(self):
        report = json.loads((DRAFT / EVIDENCE).read_bytes())
        self.assertEqual((report['planned_known_cases'], report['independently_verified_complete_cases']), (4, 3))
        self.assertFalse(report['known_batch_valid'])
        self.assertEqual(report['fresh_scenarios_executed'], 0)
        self.assertEqual([row['individual_case_pass'] for row in report['cases']], [True, True, True, None])
        self.assertNotIn('video', report['cases'][-1])
        media = [row['video'] for row in report['cases'][:3]]
        for ref in [*media, report['figure'], report['poster']]:
            blob = (DRAFT / 'assets' / ref['file']).read_bytes()
            self.assertEqual((len(blob), hashlib.sha256(blob).hexdigest()), (ref['bytes'], ref['sha256']))
